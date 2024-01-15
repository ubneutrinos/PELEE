using std::string;
using std::vector;

// Inputs:
// in_file = the input pelee nutple
// selection = the selection from the selections_str vector above you want to apply

void apply_preselection(const string in_file){

  gSystem->mkdir("noweights");

  // Get the objects from the old file
  std::cout << "Loading old trees from " << in_file << std::endl;
  TFile* f_in = TFile::Open(in_file.c_str());

  // Format the name of the output file  
  string filename = "noweights/" + in_file.substr(in_file.find_last_of("/")+1);
  TFile* f_out = new TFile(filename.c_str(),"RECREATE");
  std::cout << "Opened output file " << filename << std::endl;
  f_out->cd();
  f_out->mkdir("nuselection");
  f_out->cd("nuselection");

  std::cout << "Making clone of event tree" << std::endl;
  TTree* t_in = static_cast<TTree*>(f_in->Get("nuselection/NeutrinoSelectionFilter"));
  t_in->SetBranchStatus("weights",0);
  t_in->SetBranchStatus("weightsFlux",0);
  t_in->SetBranchStatus("weightsGenie",0);
  t_in->SetBranchStatus("weightsReint",0);


  TTree* t_out = t_in->CopyTree(queries.at(i_sel).c_str());      
  delete t_in;
  
  std::cout << "Making clone of subrun tree" << std::endl;
  TTree* sr_in = static_cast<TTree*>(f_in->Get("nuselection/SubRun"));
  TTree* sr_out = sr_in->CopyTree(""); 
  delete sr_in;
  std::cout << "Done filtering old trees" << std::endl;

  f_in->Close();

  // if slimmed weights are requested
  if(slim_weights){

    std::cout << "Changing multisim weights to unsigned shorts" << std::endl;

    TTree* t_out_slim = t_out->CloneTree(0); 

    std::vector<unsigned short>* weightsGenie = 0;
    std::vector<unsigned short>* weightsFlux = 0;
    std::vector<unsigned short>* weightsReint = 0;

    t_out->SetBranchAddress("weightsGenie",&weightsGenie); 
    t_out->SetBranchAddress("weightsFlux",&weightsFlux); 
    t_out->SetBranchAddress("weightsReint",&weightsReint); 

    for(Long64_t ientry=0;ientry<t_out->GetEntries();ientry++){
      t_out->GetEntry(ientry);
      if(ientry % 1000 == 0) std::cout << "Entry " << ientry << "/" << t_out->GetEntries() << std::endl;
      while(weightsGenie->size() > 100) weightsGenie->pop_back();
      while(weightsFlux->size() > 100) weightsFlux->pop_back();
      while(weightsReint->size() > 100) weightsReint->pop_back();
      t_out_slim->Fill();
    }    

   t_out = t_out_slim; 

  }

  std::cout << "Writing trees to output file" << std::endl;
  sr_out->Write("SubRun");
  t_out->Write("NeutrinoSelectionFilter");
  f_out->Close();

}
