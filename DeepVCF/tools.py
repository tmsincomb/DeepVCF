"""
Home for string tools and any other misc tool until its large enough for it's own file
"""
from subprocess import check_output
from typing import Union, Tuple
from pathlib import Path


def pathing(path: str, new: bool = False, overwrite: bool = True) -> Path:
    """ Guarantees correct expansion rules for pathing.

    :param Union[str, Path] path: path of folder or file you wish to expand.
    :param bool new: will check if distination exists if new  (will check parent path regardless).
    :return: A pathlib.Path object.

    >>> pathing('~/Desktop/folderofgoodstuffs/')
    /home/user/Desktop/folderofgoodstuffs
    """
    path = Path(path)
    # Expand shortened path
    if str(path)[0] == '~':
        path = path.expanduser()
    # Exand local path
    if str(path)[0] == '.':
        path = path.resolve()
    else:
        path = path.absolute()
    # Making sure new paths don't exist while also making sure existing paths actually exist.
    if new:
        if not path.parent.exists():
            raise ValueError(f'ERROR ::: Parent directory of {path} does not exist.')
        if path.exists() and not overwrite:
            raise ValueError(f'ERROR ::: {path} already exists!')
    else:
        if not path.exists():
            raise ValueError(f'ERROR ::: Path {path} does not exist.')
    return path
    

def mockreads(fasta: str, verbose=True) -> Tuple[str, str]:
    fasta = pathing(fasta)
    forward_read_path, backward_read_path = fasta.with_suffix('.read1.fq'), fasta.with_suffix('.read2.fq')
    cmd = f"wgsim -e 0 -r 0 -R 0 -X 0 -A 0 {fasta} {forward_read_path} {backward_read_path}"
    if verbose:
        print(cmd)
    _ = check_output(cmd, shell=True)
    return forward_read_path, backward_read_path