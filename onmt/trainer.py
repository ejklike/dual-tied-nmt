"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""

import torch
import traceback

import onmt.utils
from onmt.utils.logging import logger

torch.autograd.set_detect_anomaly(True)


def build_trainer(opt, device_id, model, fields, optim, model_saver=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """

    tgt_field = dict(fields)["tgt"].base_field
    train_loss = onmt.utils.loss.build_loss_compute(model, tgt_field, opt)
    valid_loss = onmt.utils.loss.build_loss_compute(
        model, tgt_field, opt, train=False)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches if opt.model_dtype == 'fp32' else 0
    norm_method = opt.normalization
    accum_count = opt.accum_count
    accum_steps = opt.accum_steps
    n_gpu = opt.world_size
    average_decay = opt.average_decay
    average_every = opt.average_every
    dropout = opt.dropout
    dropout_steps = opt.dropout_steps
    if device_id >= 0:
        gpu_rank = opt.gpu_ranks[device_id]
    else:
        gpu_rank = 0
        n_gpu = 0
    gpu_verbose_level = opt.gpu_verbose_level

    earlystopper = onmt.utils.EarlyStopping(
        opt.early_stopping, scorers=onmt.utils.scorers_from_opts(opt)) \
        if opt.early_stopping > 0 else None

    source_noise = None
    if len(opt.src_noise) > 0:
        src_field = dict(fields)["src"].base_field
        corpus_id_field = dict(fields).get("corpus_id", None)
        if corpus_id_field is not None:
            ids_to_noise = corpus_id_field.numericalize(opt.data_to_noise)
        else:
            ids_to_noise = None
        source_noise = onmt.modules.source_noise.MultiNoise(
            opt.src_noise,
            opt.src_noise_prob,
            ids_to_noise=ids_to_noise,
            pad_idx=src_field.pad_token,
            end_of_sentence_mask=src_field.end_of_sentence_mask,
            word_start_mask=src_field.word_start_mask,
            device_id=device_id
        )

    report_manager = onmt.utils.build_report_manager(opt, gpu_rank)
    trainer = onmt.Trainer(model, train_loss, valid_loss, optim, trunc_size,
                           shard_size, norm_method,
                           accum_count, accum_steps,
                           n_gpu, gpu_rank,
                           gpu_verbose_level, report_manager,
                           with_align=True if opt.lambda_align > 0 else False,
                           model_saver=model_saver if gpu_rank == 0 else None,
                           average_decay=average_decay,
                           average_every=average_every,
                           model_dtype=opt.model_dtype,
                           earlystopper=earlystopper,
                           dropout=dropout,
                           dropout_steps=dropout_steps,
                           source_noise=source_noise,
                           num_experts=opt.num_experts,
                           twoway=opt.twoway,
                           hard_selection=opt.hard_selection,
                           tgt_field=tgt_field)
    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            accum_count(list): accumulate gradients this many times.
            accum_steps(list): steps for accum gradients changes.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, model, train_loss, valid_loss, optim,
                 trunc_size=0, shard_size=32,
                 norm_method="sents", accum_count=[1],
                 accum_steps=[0],
                 n_gpu=1, gpu_rank=1, gpu_verbose_level=0,
                 report_manager=None, with_align=False, model_saver=None,
                 average_decay=0, average_every=1, model_dtype='fp32',
                 earlystopper=None, dropout=[0.3], dropout_steps=[0],
                 source_noise=None, num_experts=1, 
                 hard_selection=False, twoway=True, tgt_field=None):
        # Basic attributes.
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.norm_method = norm_method
        self.accum_count_l = accum_count
        self.accum_count = accum_count[0]
        self.accum_steps = accum_steps
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.with_align = with_align
        self.model_saver = model_saver
        self.average_decay = average_decay
        self.moving_average = None
        self.average_every = average_every
        self.model_dtype = model_dtype
        self.earlystopper = earlystopper
        self.dropout = dropout
        self.dropout_steps = dropout_steps
        self.source_noise = source_noise
        
        self.num_experts = num_experts
        self.hard_selection = hard_selection
        self.twoway = twoway
        self.tgt_field = tgt_field

        for i in range(len(self.accum_count_l)):
            assert self.accum_count_l[i] > 0
            if self.accum_count_l[i] > 1:
                assert self.trunc_size == 0, \
                    """To enable accumulated gradients,
                       you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

    def _accum_count(self, step):
        for i in range(len(self.accum_steps)):
            if step > self.accum_steps[i]:
                _accum = self.accum_count_l[i]
        return _accum

    def _maybe_update_dropout(self, step):
        for i in range(len(self.dropout_steps)):
            if step > 1 and step == self.dropout_steps[i] + 1:
                self.model.update_dropout(self.dropout[i])
                logger.info("Updated dropout to %f from step %d"
                            % (self.dropout[i], step))

    def _accum_batches(self, iterator):
        batches = []
        normalization_x2y = normalization_y2x = 0
        self.accum_count = self._accum_count(self.optim.training_step)
        for batch in iterator:
            batches.append(batch)
            if self.norm_method == "tokens":
                num_tokens = batch.tgt[0][1:, :, 0].ne(
                    self.train_loss.padding_idx).sum()
                normalization_x2y += num_tokens.item()
                num_tokens = batch.src[0][1:, :, 0].ne(
                    self.train_loss.padding_idx).sum()
                normalization_y2x += num_tokens.item()
            else:
                normalization_x2y += batch.batch_size
                normalization_y2x += batch.batch_size
            if len(batches) == self.accum_count:
                yield batches, normalization_x2y, normalization_y2x
                self.accum_count = self._accum_count(self.optim.training_step)
                batches = []
                normalization_x2y = normalization_y2x = 0
        if batches:
            yield batches, normalization_x2y, normalization_y2x

    def _update_average(self, step):
        if self.moving_average is None:
            copy_params = [params.detach().float()
                           for params in self.model.parameters()]
            self.moving_average = copy_params
        else:
            average_decay = max(self.average_decay,
                                1 - (step + 1)/(step + 10))
            for (i, avg), cpt in zip(enumerate(self.moving_average),
                                     self.model.parameters()):
                self.moving_average[i] = \
                    (1 - average_decay) * avg + \
                    cpt.detach().float() * average_decay

    def train(self,
              train_iter,
              train_steps,
              save_checkpoint_steps=5000,
              valid_iter=None,
              valid_steps=10000):
        """
        The main training loop by iterating over `train_iter` and possibly
        running validation on `valid_iter`.

        Args:
            train_iter: A generator that returns the next training batch.
            train_steps: Run training for this many iterations.
            save_checkpoint_steps: Save a checkpoint every this many
              iterations.
            valid_iter: A generator that returns the next validation batch.
            valid_steps: Run evaluation every this many iterations.

        Returns:
            The gathered statistics.
        """
        if valid_iter is None:
            logger.info('Start training loop without validation...')
        else:
            logger.info('Start training loop and validate every %d steps...',
                        valid_steps)

        total_stats = onmt.utils.Statistics(self.num_experts)
        report_stats = onmt.utils.Statistics(self.num_experts)
        self._start_report_manager(start_time=total_stats.start_time)

        for i, (batches, normalization_x2y, normalization_y2x) in enumerate(
                self._accum_batches(train_iter)):
            step = self.optim.training_step
            # UPDATE DROPOUT
            self._maybe_update_dropout(step)

            if self.gpu_verbose_level > 1:
                logger.info("GpuRank %d: index: %d", self.gpu_rank, i)
            if self.gpu_verbose_level > 0:
                logger.info("GpuRank %d: reduce_counter: %d \
                            n_minibatch %d"
                            % (self.gpu_rank, i + 1, len(batches)))

            if self.n_gpu > 1:
                normalization_x2y = sum(onmt.utils.distributed
                                    .all_gather_list
                                    (normalization_x2y))
                normalization_y2x = sum(onmt.utils.distributed
                                    .all_gather_list
                                    (normalization_y2x))

            self._gradient_accumulation(
                batches, normalization_x2y, normalization_y2x, 
                total_stats, report_stats)

            if self.average_decay > 0 and i % self.average_every == 0:
                self._update_average(step)

            report_stats = self._maybe_report_training(
                step, train_steps,
                self.optim.learning_rate(),
                report_stats)

            if valid_iter is not None and step % valid_steps == 0:
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: validate step %d'
                                % (self.gpu_rank, step))
                valid_stats = self.validate(
                    valid_iter, moving_average=self.moving_average)
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: gather valid stat \
                                step %d' % (self.gpu_rank, step))
                valid_stats = self._maybe_gather_stats(valid_stats)
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: report stat step %d'
                                % (self.gpu_rank, step))
                self._report_step(self.optim.learning_rate(),
                                  step, valid_stats=valid_stats)
                # Run patience mechanism
                if self.earlystopper is not None:
                    self.earlystopper(valid_stats, step)
                    # If the patience has reached the limit, stop training
                    if self.earlystopper.has_stopped():
                        break

            if (self.model_saver is not None
                and (save_checkpoint_steps != 0
                     and step % save_checkpoint_steps == 0)):
                self.model_saver.save(step, moving_average=self.moving_average)

            if train_steps > 0 and step >= train_steps:
                break

        if self.model_saver is not None:
            self.model_saver.save(step, moving_average=self.moving_average)
        return total_stats

    def validate(self, valid_iter, moving_average=None):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        if moving_average:
            # swap model params w/ moving average
            # (and keep the original parameters)
            model_params_data = []
            for avg, param in zip(self.moving_average,
                                  self.model.parameters()):
                model_params_data.append(param.data)
                param.data = avg.data.half() if self.optim._fp16 == "legacy" \
                    else avg.data

        # Set model in validating mode.
        self.model.eval()

        with torch.no_grad():
            stats = onmt.utils.Statistics(self.num_experts)

            for batch in valid_iter:
                valid_loss, valid_stats = self.get_loss_stats(
                    batch, validation=True)
                stats.update(valid_stats)

        if moving_average:
            for param_data, param in zip(model_params_data,
                                         self.model.parameters()):
                param.data = param_data

        # Set model back to training mode.
        self.model.train()

        return stats

    def expert_index(self, i):
        if self.num_experts > 1:
            return i + self.tgt_field.vocab.stoi['<expert_0>']
        return self.tgt_field.vocab.stoi[self.tgt_field.init_token]

    def get_one_hot_winners(self, winners):
        one_hot_winners = (winners.view(-1, 1) 
            == torch.arange(self.num_experts, device=winners.device)
               .reshape(1, self.num_experts)).float() # B x K
        return one_hot_winners

    def insert_latent_token(self, seq, winners):
        latent = seq[:1]
        latent[0, :, 0] = self.expert_index(winners)
        return torch.cat([seq[:1], latent, seq[1:]], dim=0) 

    def get_winner(self, batch, validation=False):
        src, src_lengths = batch.src
        tgt, tgt_lengths = batch.tgt

        k = self.num_experts
        bsz = src_lengths.size(0)

        if validation:
            dec_out_for_winner, _ = self.model(src, tgt[:1], src_lengths)
            winner_probs = self.model.generator(dec_out_for_winner)
            winner_probs = winner_probs[0, :, -self.num_experts:] # B x K
            winners = torch.argmax(winner_probs, dim=1)

        else:
            self.model.eval()
            with torch.no_grad():  # disable autograd
                loss_all = []

                # x2y encoding and decoder init
                src_enc, src_mem, _ = self.model.encoder(src, src_lengths)
                self.model.decoder_x2y.init_state(src, src_mem, src_enc)

                for i in range(k):
                    tgt_i = self.insert_latent_token(tgt, i)

                    dec_out, _ = self.model.decoder_x2y(
                        tgt_i[:-1], src_mem, memory_lengths=src_lengths)
                    loss_x2y, _ = self.train_loss.compute_loss(
                        dec_out, tgt_i[1:], reduced_sum=False, side='x2y')
                    loss_x2y = loss_x2y.view(bsz, -1)
                    loss_x2y = loss_x2y.sum(dim=1, keepdim=True) # -> B x 1
                    
                    # y2x
                    dec_out, _ = self.model(
                        tgt_i, src[:-1], tgt_lengths+1, side='y2x')
                    loss_y2x, _ = self.train_loss.compute_loss(
                        dec_out, src[1:], reduced_sum=False, side='y2x')
                    loss_y2x = loss_y2x.view(bsz, -1)
                    loss_y2x = loss_y2x.sum(dim=1, keepdim=True) # -> B x 1

                    loss_all.append(loss_x2y + loss_y2x)
                loss_all = torch.cat(loss_all, dim=1)  # -> B x K
                winners = loss_all.min(dim=1)[1]
            self.model.train()
            assert not winners.requires_grad

        return winners

    def get_loss_stats(self, batch, validation=False):
        src, src_lengths = batch.src
        tgt, tgt_lengths = batch.tgt

        winners = rz = None
        if self.num_experts > 1:
            winners = self.get_winner(batch, validation=validation)
            rz = self.get_one_hot_winners(winners).sum(dim=0).cpu().numpy()

            tgt = self.insert_latent_token(tgt, winners)
            tgt_lengths += 1

        dec_out, attn_x2y = self.model(src, tgt[:-1], src_lengths, side='x2y')
        loss_x2y, stat_dict_x2y = self.train_loss.compute_loss(
            dec_out, tgt[1:], side='x2y')
        
        # y2x encoding
        dec_out, attn_y2x = self.model(tgt, src[:-1], tgt_lengths, side='y2x')
        loss_y2x, stat_dict_y2x = self.train_loss.compute_loss(
            dec_out, src[1:], side='y2x')

        loss = loss_x2y + loss_y2x
        ########### if self.model.train and self.mirrored_attn:
        ########### loss += mirrored_loss....
        stats = onmt.utils.Statistics(self.num_experts, 
                                      loss=loss.item(),
                                      r=rz,
                                      **stat_dict_x2y,
                                      **stat_dict_y2x)
        return loss, stats

    def _gradient_accumulation(self, true_batches, normalization_x2y, 
                               normalization_y2x, total_stats, report_stats):
        if self.accum_count > 1:
            self.optim.zero_grad()

        for k, batch in enumerate(true_batches):
            
            batch = self.maybe_noise_source(batch)

            if self.accum_count == 1:
                self.optim.zero_grad()

            # gradient accumulation
            loss, stats = self.get_loss_stats(batch)
            self.optim.backward(loss)

            # update stats
            total_stats.update(stats)
            report_stats.update(stats)

            # Update the parameters and statistics.
            if self.accum_count == 1:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    grads = [p.grad.data for p in self.model.parameters()
                             if p.requires_grad and p.grad is not None]
                    onmt.utils.distributed.all_reduce_and_rescale_tensors(
                        grads, float(1))
                self.optim.step()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad and p.grad is not None]
                onmt.utils.distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            self.optim.step()

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return onmt.utils.Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def maybe_noise_source(self, batch):
        if self.source_noise is not None:
            return self.source_noise(batch)
        return batch
