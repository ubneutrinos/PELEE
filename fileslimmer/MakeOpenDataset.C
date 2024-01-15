// Check a PeLEE ntuple against the good runs list
// Author: C Thorpe
//
// Useage:
// On a single file CheckFile(file)
// On a list of files (set below) CheckGRList() 

const double fraction = 0.2; // fraction of events to keep

void CheckFile(std::string file,const std::string gr_list){

  gSystem->Exec("mkdir -p OpenData");

  TFile* f_in = TFile::Open(file.c_str());
  TTree* sub_tree = static_cast<TTree*>(f_in->Get("nuselection/SubRun"));
  Int_t subRun;
  sub_tree->SetBranchAddress("run",&run); 
  sub_tree->SetBranchAddress("subRun",&subRun); 

  // make a clone of the subrun tree with the duplicates removed
  TFile* f_out = new TFile(("OpenData/OpenData_" + file).c_str(),"RECREATE");
  TTree* new_sub_tree = sub_tree->CloneTree(0);
  f_out->mkdir("nuselection");
  f_out->cd("nuselection");

  // Do the same but with the events this time
  TTree* evt_tree = static_cast<TTree*>(f_in->Get("nuselection/NeutrinoSelectionFilter"));
  Int_t evt;
  evt_tree->SetBranchAddress("run",&run); 
  evt_tree->SetBranchAddress("sub",&subRun); 
  evt_tree->SetBranchAddress("evt",&evt); 

  TTree* new_evt_tree = evt_tree->CloneTree(0);

  // Use RNG to select events
  TRandom2* R = new TRandom2();

  for(int ientry=0;ientry<evt_tree->GetEntries();ientry++){

    double rn = R->Uniform(0.0,1.0);
    std::cout << rn << std::endl;   

    if(rn > fraction) continue;

    if(ientry % 20000 == 0) std::cout << "Checking event " << ientry << "/" << evt_tree->GetEntries() << std::endl;

    evt_tree->GetEntry(ientry);

    new_evt_tree->Fill();
   
    if(ientry > 1000) continue;


  } // event loop

  new_sub_tree->Write("SubRun");
  new_evt_tree->Write("NeutrinoSelectionFilter");

  f_in->Close();
  f_out->Close();

}
