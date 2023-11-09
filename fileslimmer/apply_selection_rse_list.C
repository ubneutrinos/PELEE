using std::string;
using std::vector;

void apply_selection_rse_list(const string in_file,const string rse_list){

  // Get the objects from the old file
  std::cout << "Loading old trees from " << in_file << std::endl;
  TFile* f_in = TFile::Open(in_file.c_str());
  TTree* t_in = static_cast<TTree*>(f_in->Get("nuselection/NeutrinoSelectionFilter"));
  TTree* sr_in = static_cast<TTree*>(f_in->Get("nuselection/SubRun"));
  std::cout << "Done loading old trees" << std::endl;

  // Load the rse list
  std::ifstream rse_input(rse_list);
  std::map<int,vector<std::pair<int,int>>> rse_map;
  int run,sub,evt;
  while(!rse_input.eof()){
    rse_input >> run >> sub >> evt;
    if(rse_map.find(run) == rse_map.end()) rse_map[run] = {std::make_pair(sub,evt)};
    else rse_map.at(run).push_back(std::make_pair(sub,evt));
  }


  // Format the name of the output file  
  string filename = in_file.substr(in_file.find_last_of("/")+1);
  TFile* f_out = new TFile(filename.c_str(),"RECREATE");
  std::cout << "Opened output file " << filename << std::endl;
  f_out->cd();
  f_out->mkdir("nuselection");
  f_out->cd("nuselection");

  // Make a clone of the tree with the selection applied
  std::cout << "Making clone of event tree" << std::endl;
  TTree* t_out = t_in->CloneTree();      
  t_out->Reset();
  TTree* sr_out = sr_in->CopyTree(""); 

  t_in->SetBranchAddress("run",&run); 
  t_in->SetBranchAddress("sub",&sub); 
  t_in->SetBranchAddress("evt",&evt); 

  for(int ientry=0;ientry<t_in->GetEntries();ientry++){

    if(ientry % 20000 == 0) std::cout << "Checking event " << ientry << "/" << t_in->GetEntries() << std::endl;

    t_in->GetEntry(ientry);

    //std::cout << run << " " << sub << " " << evt << std::endl;

    // only keep events with corresponding entries in the rse map 
   // if(rse_map.find(run) != rse_map.end() && std::find(rse_map.at(run).begin(),rse_map.at(run).end(),std::make_pair(sub,evt)) != rse_map.at(run).end()){
  //    t_out->Fill();      
 //   }

    if(rse_map.find(run) != rse_map.end()){
      std::vector<std::pair<int,int>>* se = &rse_map.at(run);
      std::pair<int,int> tree_se = std::make_pair(sub,evt); 
      for(size_t i_se=0;i_se<se->size();i_se++){
        if(se->at(i_se) == tree_se){
          t_out->Fill();
          se->erase(se->begin()+i_se);
          if(!se->size()) rse_map.erase(run);
          break;
        } 
      }    
    }

  }

  std::cout << "Writing trees to output file" << std::endl;
  sr_out->Write("SubRun");
  t_out->Write("NeutrinoSelectionFilter");

  f_in->Close();
  f_out->Close();

}
