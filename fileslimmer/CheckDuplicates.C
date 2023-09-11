// Check a PeLEE ntuple file for duplicated runs/subruns/events, confirm the
// run/subrun numbers match the event tree for correct POT bookkeeping
// Author: C Thorpe
//
// Useage:
// On a single file CheckFile(file)
// On all of the root files in the working dir CheckDuplicates() 

void CheckFile(std::string file){

  gSystem->Exec("mkdir -p NoDup");

  std::map<int,std::vector<int>> rs_map;

  TFile* f_in = TFile::Open(file.c_str(),"APPEND");
  TTree* sub_tree = static_cast<TTree*>(f_in->Get("nuselection/SubRun"));
  Int_t run,subRun;
  sub_tree->SetBranchAddress("run",&run); 
  sub_tree->SetBranchAddress("subRun",&subRun); 

  // make a clone of the subrun tree with the duplicates removed
  TFile* f_out = new TFile(("NoDup/" + file).c_str(),"RECREATE");
  TTree* new_sub_tree = sub_tree->CloneTree(0);
  f_out->mkdir("nuselection");
  f_out->cd("nuselection");

  int duplicates = 0;
  for(int ientry=0;ientry<sub_tree->GetEntries();ientry++){
    sub_tree->GetEntry(ientry);

    bool found_run = rs_map.find(run) != rs_map.end();
    bool found_sub = found_run && std::find(rs_map.at(run).begin(),rs_map.at(run).end(),subRun) != rs_map.at(run).end();

    if(found_run && found_sub){
      std::cout << "RS " << run << " " << subRun << " is a duplicate" << std::endl;
      duplicates++;
      continue;
    }

    if(!found_run && !found_sub) rs_map[run] = {subRun};
    else rs_map.at(run).push_back(subRun);

    new_sub_tree->Fill();

  } // subrun loop        

  std::cout << "Removed " << duplicates << " subruns" << std::endl;

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

void CheckDuplicates(){

  string lists_str = static_cast<string>(gSystem->GetFromPipe("ls | grep \".root\""));
  std::istringstream iss_lists(lists_str);

  string line;
  while(std::getline(iss_lists,line)) CheckFile(line);
  
}
