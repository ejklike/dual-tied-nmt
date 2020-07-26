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


def canonicalize_translation_output(fin, fout, aware_latent=True):
    disable_rdkit_log()

    def _get_smi_lat_from_line(line, aware_latent=False):
        line = line.strip()
        tokens = line.split(',')[0].split(' ')
        if aware_latent:
            lat = tokens[0]
            smi = ''.join(tokens[1:])
        else:
            lat = None
            smi = ''.join(tokens)
        smi_can = canonicalize_smiles(smi)
        return smi, lat

    def _tokenize(smi, lat):
        tokens = smi_tokenizer(smi)
        if lat is not None: tokens = lat + ' ' + tokens
        return tokens

    with open(fin, 'r') as f:
        lines = f.readlines()
        smi_lat_list = [
            _get_smi_lat_from_line(line, aware_latent=aware_latent) 
            for line in lines]

    smiles_can = [_tokenize(smi, lat) + '\n' for smi, lat in smi_lat_list]
    
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


def combine_translation(input_path, bwd_input_path, num_experts, beam_size, 
                        output_path=None):
    """
    Reads the output smiles from each of the latent classes and combines them.
    Args:
        input_dir: The path to the input directory containing output files
        n_latent: The number of latent classes used for the model
        beam_size: Number of smiles results per reaction
        output_path: If given, writes the combined smiles to this path
    """
    disable_rdkit_log()

    def parse(line):
        smiles, score = line.strip().split(',')
        score = float(score)
        return smiles, score

    # INPUT
    fwd_smiles_list = read_file(input_path, beam_size=beam_size,
                                parse_func=parse)
    bwd_smiles_list = read_file(bwd_input_path, beam_size=beam_size, 
                                parse_func=parse)
    # OUTPUT
    combined_list = []
    output_file = open(output_path, 'w+')
    output_file_detailed = open(output_path + '.detailed', 'w+')

    n_data = len(fwd_smiles_list)
    for data_idx in tqdm(range(n_data)):
        r_dict = {}
        output_list_fwd = fwd_smiles_list[data_idx]
        output_list_bwd = bwd_smiles_list[data_idx]

        for beam_idx, ((smiles, score), (_, score_bwd)) in \
                enumerate(zip(output_list_fwd, output_list_bwd)):
            tokens = smiles.split(' ')
            latent, smiles = tokens[0], ' '.join(tokens[1:])
            score += score_bwd
            update_condition = (
                smiles not in r_dict or   # Add the output to dictionary
                score > r_dict[smiles][0] # Update with the best score
            )
            if update_condition:
                r_dict[smiles] = (score, latent)
                
        sorted_output = sorted(r_dict.items(), 
                               key=lambda x: x[1],
                               reverse=True)
        top_smiles = []
        for beam_idx in range(beam_size):
            if beam_idx < len(sorted_output):
                smiles, (score, latent) = sorted_output[beam_idx]
                top_smiles += [smiles]
            else:
                smiles, (score, latent) = '', (-1e5, '')

            output_file.write('%s\n' % smiles)
            output_file_detailed.write('%s,%s,%.4f\n' % (smiles, latent, score))
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