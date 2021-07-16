echo "Downloading OSCAR Corpus ..."
gdown https://drive.google.com/uc?id=1k-Z31TabDEiJhFR6qEiQQwVFoo9d2Xiz
gdown https://drive.google.com/uc?id=10MpQn1VYE53PNs5KIw4htqPOiOvt6JUD
gdown https://drive.google.com/uc?id=1CFqJsy3oKMCG2KLw_uvyl71a8Bm89zeW
gdown https://drive.google.com/uc?id=1dEEKl_tG0zDxop9hCqH6Slo499ocI0fo
gdown https://drive.google.com/uc?id=1CFqJsy3oKMCG2KLw_uvyl71a8Bm89zeW
gdown https://drive.google.com/uc?id=1vp6PAhXhkVjTFxVszwBwu42W0kg9ffVa
gdown https://drive.google.com/uc?id=1oJaYEPQ-Bjj6Cuz94PWJdj8JiLbokrfk

echo "Decompressing ..."
gunzip -d ro_part_1.txt.gz
gunzip -d ro_part_2.txt.gz
gunzip -d ro_part_3.txt.gz
gunzip -d ro_part_4.txt.gz
gunzip -d ro_part_5.txt.gz
gunzip -d ro_part_6.txt.gz
ls
cat ro_part_1.txt ro_part_2.txt ro_part_3.txt ro_part_4.txt ro_part_5.txt ro_part_6.txt >ro_dedup.txt
mkdir -p raw/oscar/

mv ro_dedup.txt raw/oscar/ro_dedup.txt




