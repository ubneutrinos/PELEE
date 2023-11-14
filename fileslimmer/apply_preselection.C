using std::string;
using std::vector;

enum selections {knuepresel,knumupresel};
const vector<string> selections_str = {"nuepresel","numupresel"};
const vector<string> queries = {
  // nue preselection
  "nslice == 1 && selected == 1 && shr_energy_tot_cali > 0.07",
  // numu preselection
  "nslice == 1 && topological_score > 0.06 && reco_nu_vtx_sce_x > 5. && reco_nu_vtx_sce_x < 251. && reco_nu_vtx_sce_y > -110. && reco_nu_vtx_sce_y < 110. && reco_nu_vtx_sce_z > 20. && reco_nu_vtx_sce_z < 986."
};

// Inputs:
// in_file = the input pelee nutple
// selection = the selection from the selections_str vector above you want to apply
// slim_weights = convert the multisim weights (Genie, Flux, Reint) from floats to unsigned short to save 
// space and only keep the first 100 universes

void apply_preselection(const string in_file,const string selection,const bool slim_weights){

  int i_sel = -1;
  for(size_t i=0;i<selections_str.size();i++)
    if(selections_str.at(i) == selection) i_sel = i;

  if(i_sel == -1)
    throw std::invalid_argument("Please specify with the nue or numu preselections");

  gSystem->Exec(("mkdir -p " + selection).c_str());

  // Get the objects from the old file
  std::cout << "Loading old trees from " << in_file << std::endl;
  TFile* f_in = TFile::Open(in_file.c_str());

  // Format the name of the output file  
  string filename = selection + "/" + in_file.substr(in_file.find_last_of("/")+1);
  TFile* f_out = new TFile(filename.c_str(),"RECREATE");
  std::cout << "Opened output file " << filename << std::endl;
  f_out->cd();
  f_out->mkdir("nuselection");
  f_out->cd("nuselection");

  std::cout << "Making clone of event tree" << std::endl;
  TTree* t_in = static_cast<TTree*>(f_in->Get("nuselection/NeutrinoSelectionFilter"));
  if(slim_weights) t_in->SetBranchStatus("weights",0);
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
