/*
  mkdir noweights
  for f in `ls *.root`; do root -b -q slim_weights.C\(\"$f\"\) $f; done
 */
int slim_weights(TString filename){

  TFile f(filename);
  TTree *T = (TTree*)f.Get("nuselection/NeutrinoSelectionFilter");

  for (auto b : *(T->GetListOfBranches())) 
    {
      if (std::strncmp(b->GetName(),"weights",7)==0) {
	std::cout << "skip " << b->GetName() << std::endl;
	T->SetBranchStatus(b->GetName(), 0);
	continue;
      }
      T->SetBranchStatus(b->GetName(), 1);
    }

  TFile newfile("noweights/"+filename, "recreate");
  auto mydir = newfile.mkdir("nuselection");
  mydir->cd();
  auto newtree = T->CloneTree(-1,"fast");
  newfile.Write();

  return 0;

}
