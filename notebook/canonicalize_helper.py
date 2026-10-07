"""Runs under python3.12 (where the offline rdkit wheel actually works) via
subprocess, invoked from the main 3.13 kernel. Canonicalizes each input
SMILES to InChIKey14 the exact way the competition's scorer does (RDKit
tautomer canonicalization, then the first InChIKey block).

Usage: python3.12 canonicalize_helper.py <smiles_in.txt> <ikeys_out.txt>
One SMILES per line in, one InChIKey14 (or empty string on failure) per
line out, same order.
"""
import sys

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.inchi import MolToInchiKey
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl

logger = rkl.logger()
logger.setLevel(rkl.ERROR)
rkrb.DisableLog("rdApp.error")

enumerator = rdMolStandardize.TautomerEnumerator()

in_path, out_path = sys.argv[1], sys.argv[2]
with open(in_path) as f:
    smiles_list = [line.rstrip() for line in f]

results = []
for smiles in smiles_list:
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        results.append("")
        continue
    try:
        canon_mol = enumerator.Canonicalize(mol)
        ikey = MolToInchiKey(canon_mol)
        results.append(ikey.split("-")[0] if ikey else "")
    except Exception:
        results.append("")

with open(out_path, "w") as f:
    f.write("\n".join(results))
