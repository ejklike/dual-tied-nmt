from rdkit import Chem
from tqdm import tqdm


def disable_rdkit_log():
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')


def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return ''


def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)


def get_smiles_from_line(line, remove_score=False):
    line = line.strip()
    if remove_score:
        line = line.split(',')[0]
    smi = ''.join(line.split(' '))
    return smi


def canonicalize_translation_output(fin, fout, remove_score=True):
    with open(fin, 'r') as f:
        smiles = [get_smiles_from_line(line, remove_score=remove_score) 
                  for line in f.readlines()]

    disable_rdkit_log()
    smiles_can = [smi_tokenizer(canonicalize_smiles(smi)) + '\n' 
                  for smi in smiles]
    
    with open(fout, 'w') as f:
        f.writelines(smiles_can)
    

def tile_lines_n_times(fin, fout, n=10):
    with open(fin, 'r') as f:
        lines = f.readlines()
    with open(fout, 'w') as f:
        for line in lines:
            f.writelines([line] * n)


def read_file(file_path, beam_size=1, max_read=-1, parse_func=None):
    # read smiles file into a nested list of the size [n_data, beam]
    with open(file_path, 'r+') as f:
        lines = f.readlines()

    output_list = []  # List of beams if beam_size is > 1 else list of smiles
    cur_beam = []  # Keep track of the current beam

    for line in lines:
        if parse_func is None:
            parse = line.strip().replace(' ', '')  # default parse function
            if ',' in parse:
                # If output separated by commas, return first by default
                parse = parse.split(',')[0]
        else:
            parse = parse_func(line)

        cur_beam.append(parse)
        if len(cur_beam) == beam_size:
            if beam_size == 1:
                output_list.append(cur_beam[0])
            else:
                output_list.append(cur_beam)
            if max_read != -1 and len(output_list) >= max_read:
                break
            cur_beam = []
    return output_list


def combine_translation(input_format, num_experts, beam_size, 
                        bwd_input_format=None, bwd_target=None,
                        output_path=None, clean=False, detailed=True):
    """
    Reads the output smiles from each of the latent classes and combines them.
    Args:
        input_dir: The path to the input directory containing output files
        n_latent: The number of latent classes used for the model
        beam_size: Number of smiles results per reaction
        output_path: If given, writes the combined smiles to this path
    """
    disable_rdkit_log()
    
    # results_path is the prefix for the different latent file outputs
    expert_output_list = []
    if bwd_input_format: # get scores or smiles
        expert_output_list_bwd = []
    
    if bwd_target is not None:
        with open(bwd_target, 'r') as f:
            bwd_target = [x.strip().replace(' ', '') for x in f.readlines()]

    def parse(line):
        c_line = line.strip().replace(' ', '')
        smiles, score = c_line.split(',')
        score = float(score)
        return (smiles, score)

    for expert_id in range(num_experts):
        # input_format = '~~/fwd_out_%d.txt'
        file_path = input_format % expert_id
        smiles_list = read_file(file_path, beam_size=beam_size,
                                parse_func=parse)
        expert_output_list.append(smiles_list)
        # bwd_input_format = '~~/bwd_out_%d.txt'
        if bwd_input_format:
            file_path = bwd_input_format % expert_id
            smiles_list = read_file(file_path, beam_size=beam_size, 
                                    parse_func=parse)
            expert_output_list_bwd.append(smiles_list)

    combined_list = []

    if output_path is not None:
        output_file = open(output_path, 'w+')
        output_file_detailed = open(output_path + '.detailed', 'w+')

    n_data = len(expert_output_list[0])
    for data_idx in tqdm(range(n_data)):
        r_dict = {}
        bwd_target_smiles = bwd_target[data_idx] if bwd_target else ''
        for expert_id in range(num_experts):
            output_list = expert_output_list[expert_id][data_idx]
            if bwd_input_format:
                output_list_bwd = expert_output_list_bwd[expert_id][data_idx]
            for beam_idx, (smiles, score) in enumerate(output_list):
                if clean:
                    smiles = canonicalize_smiles(smiles)
                    if smiles == '':
                        continue
                
                bwd_smiles = ''
                if bwd_input_format:
                    bwd_smiles, bwd_score = output_list_bwd[beam_idx]
                    bwd_smiles = canonicalize_smiles(bwd_smiles)
                    score += bwd_score

                update_condition = (
                    smiles not in r_dict or   # Add the output to dictionary
                    score > r_dict[smiles][0] # Update with the best score
                )
                if update_condition:
                    r_dict[smiles] = (score, expert_id, bwd_smiles)
                
        sorted_output = sorted(r_dict.items(), 
                               key=lambda x: x[1],
                               reverse=True)
        top_smiles = []
        for beam_idx in range(beam_size):
            if beam_idx < len(sorted_output):
                smiles, (score, expert_id, bwd_smiles) = \
                    sorted_output[beam_idx]
                top_smiles += [smiles]
            else:
                smiles, (score, expert_id, bwd_smiles) = '', (-1e5, -1, '')

            cycle_correct = (1 if bwd_target_smiles == bwd_smiles else 0)

            output_file.write('%s\n' % smiles)
            output_file_detailed.write(
                '%s,%.4f,%d,%s,%d\n' 
                % (smiles, score, expert_id, bwd_smiles, cycle_correct))
        combined_list.append(top_smiles)
    if output_path is not None:
        output_file.close()
        output_file_detailed.close()
    return combined_list


# ==

def linecount(f):
    try:
        count = sum(1 for line in open(f) if line.rstrip())
    except FileNotFoundError:
        count = 0
    return count