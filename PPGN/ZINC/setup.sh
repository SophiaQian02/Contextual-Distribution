# conda activate poly_env

python equivariant_polynomials.py
mkdir -p data/ZINC/subset_poly
mkdir -p data/ZINC/subset_poly/ppgn
mkdir -p data/Alchemy/poly
mkdir -p data/Alchemy/poly/ppgn
mkdir -p data/SR/poly
mkdir -p data/SR/poly/ppgn
python dataset.py

