import os,sys

fin = open(sys.argv[-1],'r')

fname = (sys.argv[-1]).split("/")[-1]
fname = fname.split(".root")[0]
print fname

fout = open("%s.tex"%fname,'w')

err_v = []
ctr = 0
for line in fin:
    outstr = ""
    words = line.split(',')
    words = words[:-1] #[-1] = words[-1].split('\r\n')[0]
    if (ctr == 0):
        for word in words:
            outstr += ' & ' + word
        outstr += "\\\\ \hline \n"
    if (ctr > 0):
        outstr += words[0]
        for word in words[1:]:
            sys = 100.*float(word)
            outstr += ' & %.01f '%sys
        outstr += "\\\\ \n"
        tot_err = float('%.02f'%float(words[-1]))
        err_v.append(tot_err)
    fout.write(outstr)
    ctr += 1
print err_v
