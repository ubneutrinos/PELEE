
// Script for making the rs list that is fed into the POT/trigger counting tool
// Author: C Thorpe (U of Manchester)

void make_sr_list(std::string in_file){

  // Open the file and get the subrun tree
  TFile* f_in = TFile::Open(in_file.c_str());
  TTree* sr_in = static_cast<TTree*>(f_in->Get("nuselection/SubRun"));

  Int_t           run;
  Int_t           subRun;
  sr_in->SetBranchAddress("run",&run); 
  sr_in->SetBranchAddress("subRun",&subRun); 

  // make a text file to write the output
  gSystem->Exec("mkdir -p rs_lists");
  string filename =  "rs_lists/" + in_file.substr(in_file.find_last_of("/")+1) + ".txt";
  std::ofstream output(filename);

  // Get all of the runs/subruns in the file
  for(Long64_t ientry=0;ientry<sr_in->GetEntries();ientry++){
    sr_in->GetEntry(ientry);
    output << run << " " << subRun << std::endl;
  }

  f_in->Close();
  output.close();

}
