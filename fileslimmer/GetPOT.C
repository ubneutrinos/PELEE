
void GetPOT(std::string file){

  TFile* f_in = TFile::Open(file.c_str());
  TTree* treein = (TTree*)f_in->Get("/nuselection/SubRun");

  Float_t         pot;
  treein->SetBranchAddress("pot",&pot);

  double totalpot = 0.0;
  for(size_t ientry=0;ientry<treein->GetEntries();ientry++){
    if(ientry % 5000 == 0) std::cout << "Event " << ientry << "/" << treein->GetEntries() << std::endl;
    treein->GetEntry(ientry);
    totalpot += pot;
  }

  std::cout << "POT=" << totalpot << std::endl;


}
