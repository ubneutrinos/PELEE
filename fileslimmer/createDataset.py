import sys
import os

#print("The arguments are: "+str(sys.argv))

if len(sys.argv)!=2: exit(1)

fn = str(sys.argv[1])

tag = fn.replace('list','').replace('.txt','')
#print(tag)

dn = "cerati"+tag

cmd = "samweb create-definition "
cmd+=dn
cmd+=" \""
#BNB
cmd+="(defname:data_bnb_mcc9.1_v08_00_00_25_reco2_C1_beam_good_reco2 or defname:data_bnb_mcc9.1_v08_00_00_25_reco2_D1_beam_good_reco2 or defname:data_bnb_mcc9.1_v08_00_00_25_reco2_D2_beam_good_reco2 or defname:data_bnb_mcc9.1_v08_00_00_25_reco2_E1_beam_good_reco2 or defname:data_bnb_mcc9.1_v08_00_00_25_reco2_F_beam_good_reco2 or defname:data_bnb_mcc9.1_v08_00_00_25_reco2_G1_beam_good_reco2) and (run_number "

cnt = 0
f = open(fn,'r')
for rs in f:
    rs = rs.rstrip()
    if '.' not in rs: continue
    if cnt==0 :
        cmd += rs
    else: 
        cmd += ", "+rs
    cnt+=1

cmd+=")\""

print(cmd)

os.system(cmd)

print("created dataset with name "+dn+" containing "+str(cnt)+" events from input file: "+fn)
