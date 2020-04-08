#!/usr/bin/env python
import argparse
import torch


def average_models(model_files, fp32=False):
    vocab = None
    opt = None
    avg_model = None
    avg_generator = None

    for i, model_file in enumerate(model_files):
        m = torch.load(model_file, map_location='cpu')
        model_weights = m['model']
        generator_weights = m['generator']
        prior_x2y_weights = prior_y2x_weights = None
        if m.get('prior_x2y', None):
            prior_x2y_weights = m['prior_x2y']
        if m.get('prior_y2x', None):
            prior_y2x_weights = m['prior_y2x']

        if fp32:
            for k, v in model_weights.items():
                model_weights[k] = v.float()
            for k, v in generator_weights.items():
                generator_weights[k] = v.float()
            if prior_x2y_weights is not None:
                for k, v in prior_x2y_weights.items():
                    prior_x2y_weights[k] = v.float()
            if prior_y2x_weights is not None:
                for k, v in prior_y2x_weights.items():
                    prior_y2x_weights[k] = v.float()

        if i == 0:
            vocab, opt = m['vocab'], m['opt']
            avg_model = model_weights
            avg_generator = generator_weights
            avg_prior_x2y = prior_x2y_weights if prior_x2y_weights else None
            avg_prior_y2x = prior_y2x_weights if prior_y2x_weights else None
        else:
            for (k, v) in avg_model.items():
                avg_model[k].mul_(i).add_(model_weights[k]).div_(i + 1)

            for (k, v) in avg_generator.items():
                avg_generator[k].mul_(i).add_(generator_weights[k]).div_(i + 1)

            if prior_x2y_weights is not None:
                for (k, v) in avg_prior_x2y.items():
                    avg_prior_x2y[k].mul_(i).add_(prior_x2y_weights[k])\
                        .div_(i + 1)
            
            if prior_y2x_weights is not None:
                for (k, v) in avg_prior_y2x.items():
                    avg_prior_y2x[k].mul_(i).add_(prior_y2x_weights[k])\
                        .div_(i + 1)

    final = {"vocab": vocab, "opt": opt, "optim": None,
             "generator": avg_generator, "model": avg_model,
             "prior_x2y": avg_prior_x2y, "prior_y2x": avg_prior_y2x}
    return final


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-models", "-m", nargs="+", required=True,
                        help="List of models")
    parser.add_argument("-output", "-o", required=True,
                        help="Output file")
    parser.add_argument("-fp32", "-f", action="store_true",
                        help="Cast params to float32")
    opt = parser.parse_args()

    final = average_models(opt.models, opt.fp32)
    torch.save(final, opt.output)


if __name__ == "__main__":
    main()
