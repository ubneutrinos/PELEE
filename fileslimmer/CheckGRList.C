// Check a PeLEE ntuple against the good runs list
// Author: C Thorpe
//
// Useage:
// On a single file CheckFile(file)
// On a list of files (set below) CheckGRList() 

void CheckFile(std::string file,const std::string gr_list){

  // load the good run list from file
  std::ifstream in_gr_list(gr_list);
  std::vector<int> gr_list_v;
  int run;
  while(!in_gr_list.eof()){
    in_gr_list >> run;
    gr_list_v.push_back(run);
  }
  std::cout << "Loaded " << gr_list_v.size() << " good runs" << std::endl;

  gSystem->Exec("mkdir -p GoodRuns");

  std::map<int,std::vector<int>> rs_map;

  TFile* f_in = TFile::Open(file.c_str());
  TTree* sub_tree = static_cast<TTree*>(f_in->Get("nuselection/SubRun"));
  Int_t subRun;
  sub_tree->SetBranchAddress("run",&run); 
  sub_tree->SetBranchAddress("subRun",&subRun); 

  // make a clone of the subrun tree with the duplicates removed
  TFile* f_out = new TFile(("GoodRuns/" + file).c_str(),"RECREATE");
  TTree* new_sub_tree = sub_tree->CloneTree(0);
  f_out->mkdir("nuselection");
  f_out->cd("nuselection");

  int bad = 0;
  for(int ientry=0;ientry<sub_tree->GetEntries();ientry++){
    sub_tree->GetEntry(ientry);

    if(std::find(gr_list_v.begin(),gr_list_v.end(),run) == gr_list_v.end()){
      //std::cout << "RS " << run << " " << subRun << " is bad" << std::endl;
      bad++;
      continue;
    }

    if(rs_map.find(run) == rs_map.end()) rs_map[run] = {subRun};
    else rs_map.at(run).push_back(subRun);

    new_sub_tree->Fill();

  } // subrun loop        

  std::cout << "Removed " << bad << " subruns out of " << sub_tree->GetEntries() << std::endl;

  // Do the same but with the events this time
  TTree* evt_tree = static_cast<TTree*>(f_in->Get("nuselection/NeutrinoSelectionFilter"));
  Int_t evt;
  evt_tree->SetBranchAddress("run",&run); 
  evt_tree->SetBranchAddress("sub",&subRun); 
  evt_tree->SetBranchAddress("evt",&evt); 

  TTree* new_evt_tree = evt_tree->CloneTree(0);

  std::map<std::pair<int,int>,std::vector<int>> rse_map;

  for(int ientry=0;ientry<evt_tree->GetEntries();ientry++){

    if(ientry % 20000 == 0) std::cout << "Checking event " << ientry << "/" << evt_tree->GetEntries() << std::endl;

    evt_tree->GetEntry(ientry);

    // only keep events with corresponding entries in the rs tree    
    if(rs_map.find(run) != rs_map.end() && std::find(rs_map.at(run).begin(),rs_map.at(run).end(),subRun) != rs_map.at(run).end()){

      std::pair<int,int> rs = std::make_pair(run,subRun);               

      bool found_rs = rse_map.find(rs) != rse_map.end();
      bool found_evt = found_rs && std::find(rse_map.at(rs).begin(),rse_map.at(rs).end(),evt) != rse_map.at(rs).end();

      // check this event is a duplicate
      if(found_rs && found_evt){
        std::cout << "RSE " << run << " " << subRun << " " << evt << " is a duplicate" << std::endl;
        continue;
      }

      if(!found_rs && !found_evt) rse_map[rs] = {evt};
      else rse_map.at(rs).push_back(evt);

      new_evt_tree->Fill();

    }

  } // event loop

  new_sub_tree->Write("SubRun");
  new_evt_tree->Write("NeutrinoSelectionFilter");

  f_in->Close();
  f_out->Close();

}

std::string list_file = "pass_r4.txt";
std::vector<std::string> files = {
"ShowerSideband_Run4c_bnb_beamOn_PeLee_ntuples_run4c_ana_ana.root",
"SignalRemoved_Run4c_bnb_beamOn_PeLee_ntuples_run4c_ana_ana.root",
"TwoShowers_Run4c_bnb_beamOn_PeLee_ntuples_run4c_ana_ana.root",
"NuMuSideband_Run4c_bnb_beamOn_PeLee_ntuples_run4c_ana_ana.root",
"Run4c_bnb_beamOn_PeLee_ntuples_run4c_ana_ana.root",
"Run4c_bnb_beamOff_PeLee_ntuples_run4c_ana_ana.root"
};

void CheckGRList(){

  for(size_t i_f=0;i_f<files.size();i_f++) CheckFile(files.at(i_f),list_file); 

}
