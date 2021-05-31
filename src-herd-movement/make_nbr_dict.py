import json
from neighbor_dict import company_list
p1 = "/media/anishmm/anish-ext-disk/cs/iitbombay/sem2/stock/code/third_party/dtaidistance-master/nbr_dict.json"

with open(p1, "r") as f:
    all_nbrs_list = json.load(f)

# print(all_nbrs_list["AAPL"])

nbr_dict = dict()
for c in company_list:
    for j in all_nbrs_list[c]:
        if j in company_list and c != j:
            nbr_dict[c] = j
            break


with open("./reduced_nbr_dict.json", "w+") as f:
    json.dump(nbr_dict, f)
