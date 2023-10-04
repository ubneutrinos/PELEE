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

void apply_preselection(const string in_file,const string selection){

  int i_sel = -1;
  for(size_t i=0;i<selections_str.size();i++)
    if(selections_str.at(i) == selection) i_sel = i;

  if(i_sel == -1)
    throw std::invalid_argument("Please specify with the nue or numu preselections");

  gSystem->Exec(("mkdir -p " + selection).c_str());

  // Get the objects from the old file
  std::cout << "Loading old trees from " << in_file << std::endl;
  TFile* f_in = TFile::Open(in_file.c_str());
  TTree* t_in = static_cast<TTree*>(f_in->Get("nuselection/NeutrinoSelectionFilter"));
  TTree* sr_in = static_cast<TTree*>(f_in->Get("nuselection/SubRun"));
  std::cout << "Done loading old trees" << std::endl;

  // Format the name of the output file  
  string filename = selection + "/" + in_file.substr(in_file.find_last_of("/")+1);
  TFile* f_out = new TFile(filename.c_str(),"RECREATE");
  std::cout << "Opened output file " << filename << std::endl;
  f_out->cd();
  f_out->mkdir("nuselection");
  f_out->cd("nuselection");

  // Make a clone of the tree with the selection applied
  std::cout << "Making clone of event tree" << std::endl;
  TTree* t_out = t_in->CopyTree(queries.at(i_sel).c_str());      

  std::cout << "Writing trees to output file" << std::endl;
  sr_in->Write("SubRun");
  t_out->Write("NeutrinoSelectionFilter");

  f_in->Close();
  f_out->Close();

}
