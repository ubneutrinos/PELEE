#include "Riostream.h"

void slimmer(TString fname)
{


  
   // Get old file, old tree and set top branch address
   TString dir = "/home/david/data/searchingfornues/v08_00_00_25/cc0pinp/1115/";
   TString fullpath = dir + fname + ".root";
   TString textpath = dir + fname + ".txt";
   TString foutname = dir + fname + "_sbnfit" + ".root";
   gSystem->ExpandPathName(dir);
   //const auto filename = gSystem->AccessPathName(dir) ? "./Event.root" : "$ROOTSYS/test/Event.root";
   TFile oldfile(fullpath);
   TTree *oldtree;
   oldfile.GetObject("nuselection/NeutrinoSelectionFilter", oldtree);

  // load input text file with event/subrun/run
  ifstream infile;
  infile.open(textpath);

  int runf,subf,evtf;
  
   const auto nentries = oldtree->GetEntries();

   // Deactivate all branches
   oldtree->SetBranchStatus("*", 0);
   // Activate only four of them
   for (auto activeBranchName : {"run","weights","shr_energy_tot","slpdg","nu_e","nslice","selected","NeutrinoEnergy2","crtveto","crthitpe","trk_len",
	 "_closestNuCosmicDist","topological_score","nu_pdg","leeweight",
	 "reco_nu_vtx_sce_x","reco_nu_vtx_sce_y","reco_nu_vtx_sce_z","n_showers_contained","hits_y","hits_ratio","CosmicIP","shr_distance","tksh_distance","trk_distance",
	 "tksh_angle","shr_tkfit_dedx_Y","shr_score","trk_score","slclustfrac","trk_chipr","shrsubclusters0","shrsubclusters1","shrsubclusters2",
	 "run","sub","evt"
	 })
      oldtree->SetBranchStatus(activeBranchName, 1);


   int run,sub,evt;
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
   float trk_chipr;
     
   
   //std::map<std::string,std::vector<float>> weightsMap;
   //Event *event = nullptr;

   oldtree->SetBranchAddress("run", &run);
   oldtree->SetBranchAddress("sub", &evt);
   oldtree->SetBranchAddress("evt", &evt);
   
   //oldtree->SetBranchAddress("bdt_global", &bdt_global);
   oldtree->SetBranchAddress("nslice", &nslice);
   oldtree->SetBranchAddress("nu_pdg", &nu_pdg);
   oldtree->SetBranchAddress("selected", &selected);
   oldtree->SetBranchAddress("selected", &selected);
   oldtree->SetBranchAddress("_closestNuCosmicDist",&_closestNuCosmicDist);
   oldtree->SetBranchAddress("topological_score",&topological_score);
   oldtree->SetBranchAddress("trk_len",&trk_len);
   oldtree->SetBranchAddress("crthitpe",&crthitpe);
   oldtree->SetBranchAddress("crtveto",&crtveto);
   
   // nue variables
   oldtree->SetBranchAddress("leeweight",&leeweight);
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
   
   // Create a new file + a clone of old tree in new file
   TFile newfile(foutname, "recreate");
   auto newtree = oldtree->CloneTree(0);
   for (auto i : ROOT::TSeqI(nentries)) {
      oldtree->GetEntry(i);

      // for text-file based selection
      /*
      // is this event in the text file of selected events?
      infile.clear();
      infile.seekg(0,ios::beg);
      bool foundevent = false;
      while (1) {
	infile >> runf >> subf >> evtf;
	if (!infile.good()) break;
	if ( (run != runf) || (sub != subf) || (evt != evtf) )
	  continue;
	foundevent = true;
	//printf("run = %i, sub = %i, evt = %i",run,sub,evt);
      }

      if (foundevent == false) continue;
      */
      
      
      if (fabs(nu_pdg) == 12) continue;
      if ( (nslice == 1) &&  (crtveto !=1) && (_closestNuCosmicDist > 20.) && (topological_score > 0.06) && (trk_len > 20.) &&
	   (reco_nu_vtx_sce_x > 5.) && (reco_nu_vtx_sce_x < 251.) && (reco_nu_vtx_sce_y > -110) && (reco_nu_vtx_sce_y < 110) && (reco_nu_vtx_sce_z > 20.) && (reco_nu_vtx_sce_z < 986) ) {
	
      
      newtree->Fill();

      }

   }// for all entries
   newtree->Print();
   newfile.Write();
}
