#include "Riostream.h"
#include <map>

void slimmer_numu(TString fname)
{


  
   // Get old file, old tree and set top branch address
   TString dir = "/home/david/data/searchingfornues/v08_00_00_33/cc0pinp/0109/";
   TString fullpath = dir + fname + ".root";
   TString textpath = dir + "txt/" + fname + "_numu.txt";
   TString foutname = dir + fname + "_numu_sbnfit" + ".root";
   gSystem->ExpandPathName(dir);
   //const auto filename = gSystem->AccessPathName(dir) ? "./Event.root" : "$ROOTSYS/test/Event.root";
   TFile oldfile(fullpath);
   TTree *oldtree;
   oldfile.GetObject("nuselection/NeutrinoSelectionFilter", oldtree);

  // load input text file with event/subrun/run
  ifstream infile;
  infile.open(textpath);

  int runf,subf,evtf;
  float muonenergyf, neutrinoenergyf, muonanglef;

  // 2nd map links event to <muon angle, muon energy, neutrino energy>
  std::map<int, std::vector< std::pair<int, std::vector<float> > > > run_event_map;

  // for text-file based selection
  // is this event in the text file of selected events?
  infile.clear();
  infile.seekg(0,ios::beg);
  bool foundevent = false;
  int nlines = 0;
  
  while (1) {
    if (!infile.good()) break;
    nlines += 1;
    infile >> runf >> subf >> evtf >> muonanglef >> muonenergyf >> neutrinoenergyf;
    if (run_event_map.find(runf) == run_event_map.end()) {

      std::vector<float> muoninfo{ muonanglef, muonenergyf, neutrinoenergyf };
      std::pair<int, std::vector<float> > eventinfo = std::make_pair( evtf, muoninfo );
      std::vector< std::pair<int, std::vector<float> > > evt_v{eventinfo};
      
      run_event_map[runf] = evt_v;
    }
    else {

      std::vector<float> muoninfo{ muonanglef, muonenergyf, neutrinoenergyf };
      std::pair<int, std::vector<float> > eventinfo = std::make_pair( evtf, muoninfo );
            
      run_event_map[runf].push_back( eventinfo );
    }
  }
  
  const auto nentries = oldtree->GetEntries();

   // Deactivate all branches
   oldtree->SetBranchStatus("*", 0);
   // Activate only four of them
   for (auto activeBranchName : {"run","weights","nu_e","nslice","selected"
	 "nu_pdg","leeweight","weightSpline","weightTune","weightSplineTimesTune",
	 "run","sub","evt","npi0","category","ccnc"
	 })
      oldtree->SetBranchStatus(activeBranchName, 1);


   float weightSpline;
   int run,sub,evt;
   int ccnc;
   int nslice;
   int nu_pdg;
   int selected;
   int crtveto;
   float leeweight;
   float crthitpe;
   float trk_len;
   double _closestNuCosmicDist;
   float topological_score;

   // nue variables
   float reco_nu_vtx_sce_x, reco_nu_vtx_sce_y, reco_nu_vtx_sce_z;
   unsigned int n_showers_contained;
   float shr_energy_tot;
   unsigned int hits_y;
   float hits_ratio;
   float CosmicIP;
   float shr_distance, trk_distance, tksh_distance;
   float tksh_angle;
   float shr_tkfit_dedx_Y;
   float shr_score, trk_score;
   float slclustfrac;
   unsigned int shrsubclusters0, shrsubclusters1, shrsubclusters2;
   float shr_energy_tot_cali;
   float trk_energy_tot;
   float trk_chipr;
   int npi0, category;
   float NeutrinoEnergy2, trk_theta, trk_energy_muon;


   int numberofeventspass = 0;
   
   //std::map<std::string,std::vector<float>> weightsMap;
   //Event *event = nullptr;

   oldtree->SetBranchAddress("run", &run);
   oldtree->SetBranchAddress("sub", &evt);
   oldtree->SetBranchAddress("evt", &evt);
   
   //oldtree->SetBranchAddress("bdt_global", &bdt_global);
   oldtree->SetBranchAddress("nslice", &nslice);
   /*
   oldtree->SetBranchAddress("nu_pdg", &nu_pdg);
   oldtree->SetBranchAddress("selected", &selected);
   oldtree->SetBranchAddress("selected", &selected);
   oldtree->SetBranchAddress("_closestNuCosmicDist",&_closestNuCosmicDist);
   oldtree->SetBranchAddress("topological_score",&topological_score);
   oldtree->SetBranchAddress("trk_len",&trk_len);
   oldtree->SetBranchAddress("crthitpe",&crthitpe);
   oldtree->SetBranchAddress("crtveto",&crtveto);
   oldtree->SetBranchAddress("category",&category);
   oldtree->SetBranchAddress("npi0",&npi0);
   oldtree->SetBranchAddress("ccnc",&ccnc);
   
   // nue variables
   oldtree->SetBranchAddress("leeweight",&leeweight);
   oldtree->SetBranchAddress("weightSpline",&weightSpline);
   oldtree->SetBranchAddress("reco_nu_vtx_sce_x",&reco_nu_vtx_sce_x);
   oldtree->SetBranchAddress("reco_nu_vtx_sce_y",&reco_nu_vtx_sce_y);
   oldtree->SetBranchAddress("reco_nu_vtx_sce_z",&reco_nu_vtx_sce_z);
   oldtree->SetBranchAddress("n_showers_contained",&n_showers_contained);
   oldtree->SetBranchAddress("shr_energy_tot",&shr_energy_tot);
   oldtree->SetBranchAddress("hits_y",&hits_y);
   oldtree->SetBranchAddress("hits_ratio",&hits_ratio);
   oldtree->SetBranchAddress("CosmicIP",&CosmicIP);
   oldtree->SetBranchAddress("shr_distance",&shr_distance);
   oldtree->SetBranchAddress("trk_distance",&trk_distance);
   oldtree->SetBranchAddress("tksh_distance",&tksh_distance);
   oldtree->SetBranchAddress("tksh_angle",&tksh_angle);
   oldtree->SetBranchAddress("shr_tkfit_dedx_Y",&shr_tkfit_dedx_Y);
   oldtree->SetBranchAddress("shr_score",&shr_score);
   oldtree->SetBranchAddress("trk_score",&trk_score);
   oldtree->SetBranchAddress("trk_chipr",&trk_chipr);
   oldtree->SetBranchAddress("slclustfrac",&slclustfrac);
   oldtree->SetBranchAddress("shrsubclusters0",&shrsubclusters0);
   oldtree->SetBranchAddress("shrsubclusters1",&shrsubclusters1);
   oldtree->SetBranchAddress("shrsubclusters2",&shrsubclusters2);
   oldtree->SetBranchAddress("shr_energy_tot_cali",&shr_energy_tot_cali);
   oldtree->SetBranchAddress("trk_energy_tot",&trk_energy_tot);
   oldtree->SetBranchAddress("NeutrinoEnergy2",&NeutrinoEnergy2);
   oldtree->SetBranchAddress("trk_energy_muon",&trk_energy_muon);
   oldtree->SetBranchAddress("trk_theta",&trk_theta);
   */

   // new branch with weight = leeweight * weightSpline
   float eventweight;
   float reco_e;
   float muonangle, muonenergy, neutrinoenergy;
   
   // Create a new file + a clone of old tree in new file
   TFile newfile(foutname, "recreate");
   auto newtree = oldtree->CloneTree(0);
   newtree->Branch("eventweight",&eventweight,"eventweight/F");
   newtree->Branch("reco_e",&reco_e,"reco_e/F");
   newtree->Branch("muonangle",&muonangle,"muonangle/F");
   newtree->Branch("muonenergy",&muonenergy,"muonenergy/F");
   newtree->Branch("neutrinoenergy",&neutrinoenergy,"neutrinoenergy/F");
   
   for (auto i : ROOT::TSeqI(nentries)) {
      oldtree->GetEntry(i);


      if (nslice == 1) { // &&  
	//(reco_nu_vtx_sce_x > 5.) && (reco_nu_vtx_sce_x < 251.) && (reco_nu_vtx_sce_y > -110) && (reco_nu_vtx_sce_y < 110) && (reco_nu_vtx_sce_z > 20.) && (reco_nu_vtx_sce_z < 986) &&
	//  ( (reco_nu_vtx_sce_z < 675.) || (reco_nu_vtx_sce_z > 775.) ) &&
	//  ( topological_score> 0.06) ) { //&&
	//(crtveto!=1 || crthitpe < 100.) && (_closestNuCosmicDist > 5.) ) {
	
	eventweight = leeweight * weightSpline;
	reco_e = NeutrinoEnergy2/1000. + 0.105; // ((shr_energy_tot_cali+0.030)/0.79) + trk_energy_tot;

	if (i % 1000 == 0) 
	  printf("new event. run : %i evt : %i \n",run,evt);
	
	// find in run/event map
	if (run_event_map.find(run) == run_event_map.end())
	  continue;

	auto eventinfo_v = run_event_map[run];

	bool found = false;
	
	for (size_t ne=0; ne < eventinfo_v.size(); ne++) {

	  auto eventinfo = eventinfo_v[ne];

	  int evtf = eventinfo.first;
	  auto muoninfo = eventinfo.second;
	  muonangle      = muoninfo[0];
	  muonenergy     = muoninfo[1];
	  neutrinoenergy = muoninfo[2];
	  
	  if (evt == evtf) {
	    found = true;
	    break;
	  }
	}
	
	if (found == false) continue;

	//printf("\t found! \n");
	
	newtree->Fill();
	
	
      }// if cuts pass
      
   }// for all entries
   newtree->Print();
   newfile.Write();
}
