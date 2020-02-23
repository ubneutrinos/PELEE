#include "Riostream.h"
#include <map>

void slimmer(TString fname,float splinexsecshift=0.)
{


  
   // Get old file, old tree and set top branch address
   TString dir = "/home/david/data/searchingfornues/v08_00_00_33/cc0pinp/0109/";
   TString fullpath = dir + fname + ".root";
   TString textpath = dir + "txt/detsys/" + fname + ".txt";
   TString foutname = dir + "SBNFit/" + fname + "_1eNp_sbnfit_detsys" + ".root";
   gSystem->ExpandPathName(dir);
   //const auto filename = gSystem->AccessPathName(dir) ? "./Event.root" : "$ROOTSYS/test/Event.root";
   TFile oldfile(fullpath);
   TTree *oldtree;
   oldfile.GetObject("nuselection/NeutrinoSelectionFilter", oldtree);


   // load MCC8 SPLINE XSEC
   TString splinepath = "/home/david/Downloads/ccqe_spline_ratios.root";
   TFile splines(splinepath);
   TGraph *MCC9spline;
   splines.GetObject("nu_e_ccqe_v304a", MCC9spline);
   double energy,xsec;
   size_t nsplinepoints = MCC9spline->GetN();
   std::vector<float> spline_energy_v;
   std::vector<float> spline_xsec_v;
   for (size_t n=0; n < nsplinepoints; n++) {
     MCC9spline->GetPoint(n,energy,xsec);
     if (energy > 2.5) continue;
     spline_energy_v.push_back( energy );
     spline_xsec_v.push_back( xsec );
     //printf("xsec @ energy %f is %f\n",energy,xsec);     
   }
   float spline_binwidth = (spline_energy_v[1] - spline_energy_v[0]);
   int binshift = int(splinexsecshift / spline_binwidth);
   printf("spline bin width is %f \n",spline_binwidth);
   for (int n=0; n < spline_energy_v.size(); n++) {
     if (n%10 != 0) continue;
     float xsecratio = 1.;
     if ( (n+binshift >= 0) && (n+binshift < spline_energy_v.size())) {
       if (spline_xsec_v[n] < 1e-5)
	 xsecratio = 1e3;
       else if (spline_xsec_v[n+binshift] < 1e-5)
	 xsecratio = 0.;
       else
	 xsecratio = spline_xsec_v[n+binshift]/spline_xsec_v[n];
     }
     printf("the bin shift %f at energy %f is %f \n",splinexsecshift,spline_energy_v[n],xsecratio);
   }


   // load input text file with event/subrun/run
  ifstream infile;
  infile.open(textpath);

  int runf,subf,evtf;

  std::map<int,std::vector<int>> run_event_map;

  // for text-file based selection
  // is this event in the text file of selected events?
  infile.clear();
  infile.seekg(0,ios::beg);
  bool foundevent = false;
  int nlines = 0;
  
  while (1) {
    if (!infile.good()) break;
    nlines += 1;
    infile >> runf >> subf >> evtf;
    if (run_event_map.find(runf) == run_event_map.end()) {
      std::vector<int> evt_v = {evtf};
      run_event_map[runf] = evt_v;
    }
    else {
      run_event_map[runf].push_back( evtf );
    }
  }

  printf("there are %i lines \n",nlines);
  
   const auto nentries = oldtree->GetEntries();

   // Deactivate all branches
   oldtree->SetBranchStatus("*", 0);
   // Activate only four of them
   for (auto activeBranchName : {"run","weights","shr_energy_tot","slpdg","nu_e","nslice","selected","NeutrinoEnergy2","crtveto","crthitpe","trk_len",
	 "_closestNuCosmicDist","topological_score","nu_pdg","leeweight","weightSpline","weightTune","weightSplineTimesTune",
	 "reco_nu_vtx_sce_x","reco_nu_vtx_sce_y","reco_nu_vtx_sce_z","n_showers_contained","hits_y","hits_ratio","CosmicIP","shr_distance","tksh_distance","trk_distance",
	 "tksh_angle","shr_tkfit_dedx_Y","shr_score","trk_score","slclustfrac","trk_chipr","shrsubclusters0","shrsubclusters1","shrsubclusters2","shr_energy_tot_cali","trk_energy_tot",
	 "run","sub","evt","npi0","category","ccnc","interaction"
	 })
      oldtree->SetBranchStatus(activeBranchName, 1);


   float weightSpline;
   int run,sub,evt;
   int interaction;
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
   float nu_e, NeutrinoEnergy2;
     
   
   //std::map<std::string,std::vector<float>> weightsMap;
   //Event *event = nullptr;

   oldtree->SetBranchAddress("run", &run);
   oldtree->SetBranchAddress("sub", &evt);
   oldtree->SetBranchAddress("evt", &evt);
   oldtree->SetBranchAddress("interaction", &interaction);
   oldtree->SetBranchAddress("ccnc",&ccnc);
   oldtree->SetBranchAddress("nu_e",&nu_e);
   //oldtree->SetBranchAddress("category",&category);
   
   /*
   //oldtree->SetBranchAddress("bdt_global", &bdt_global);
   oldtree->SetBranchAddress("nslice", &nslice);
   oldtree->SetBranchAddress("nu_pdg", &nu_pdg);
   oldtree->SetBranchAddress("selected", &selected);
   oldtree->SetBranchAddress("selected", &selected);
   oldtree->SetBranchAddress("_closestNuCosmicDist",&_closestNuCosmicDist);
   oldtree->SetBranchAddress("topological_score",&topological_score);
   oldtree->SetBranchAddress("trk_len",&trk_len);
   oldtree->SetBranchAddress("NeutrinoEnergy2",&NeutrinoEnergy2);
   oldtree->SetBranchAddress("crthitpe",&crthitpe);
   oldtree->SetBranchAddress("crtveto",&crtveto);
   oldtree->SetBranchAddress("category",&category);
   oldtree->SetBranchAddress("npi0",&npi0);
   oldtree->SetBranchAddress("ccnc",&ccnc);
   
   // nue variables
   oldtree->SetBranchAddress("nu_e",&nu_e);
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
   */

   oldtree->SetBranchAddress("trk_len",&trk_len);
   oldtree->SetBranchAddress("NeutrinoEnergy2",&NeutrinoEnergy2);
   oldtree->SetBranchAddress("shr_energy_tot_cali",&shr_energy_tot_cali);
   oldtree->SetBranchAddress("trk_energy_tot",&trk_energy_tot);

   // new branch with weight = leeweight * weightSpline
   float eventweight;
   float reco_e;
   double mcc8weight; // scaling for xsec shift
   
   // Create a new file + a clone of old tree in new file
   TFile newfile(foutname, "recreate");
   auto newtree = oldtree->CloneTree(0);
   //newtree->Branch("eventweight",&eventweight,"eventweight/F");
   //newtree->Branch("reco_e",&reco_e,"reco_e/F");
   
   double nu_e_d, NeutrinoEnergy2_d, reco_e_d,trk_len_d;
   newtree->Branch("nu_e_d",&nu_e_d,"nu_e_d/D");
   newtree->Branch("NeutrinoEnergy2_d",&NeutrinoEnergy2_d,"NeutrinoEnergy2_d/D");
   newtree->Branch("reco_e_d",&reco_e_d,"reco_e_d/D");
   newtree->Branch("trk_len_d",&trk_len_d,"trk_len_d/D");
   newtree->Branch("mcc8weight",&mcc8weight,"mcc8weight/D");
   
   for (auto i : ROOT::TSeqI(nentries)) {
      oldtree->GetEntry(i);

      //eventweight = leeweight * weightSpline;
      reco_e = ((shr_energy_tot_cali+0.030)/0.79) + trk_energy_tot;


      nu_e_d = nu_e;
      NeutrinoEnergy2_d = NeutrinoEnergy2;
      reco_e_d = reco_e;
      trk_len_d = trk_len;
      
      // for numu files
      //if (fabs(nu_pdg) == 12) continue;
      //if ( (npi0 == 1) && (category != 5)) continue;

      // for pi0 files
      //if (category == 5) continue;

      //printf("new event. run : %i evt : %i \n",run,evt);
      
      // find in run/event map
      if (run_event_map.find(run) == run_event_map.end())
	continue;
      
      auto evt_v = run_event_map[run];
      
      bool found = false;
      for (size_t ne=0; ne < evt_v.size(); ne++) {
	if (evt == evt_v[ne]) {
	    found = true;
	    break;
	}
      }
      
      if (found == false) continue;

      // set MCC8 weight
      mcc8weight = 1.0;
      if ( (interaction == 0) && (ccnc == 0) && (nu_e < 2.5) ) {
	// energy bin for xsec ratio:
	int ebin = int(nu_e/spline_binwidth);
	if ( (ebin+binshift >= 0) && (ebin+binshift < spline_energy_v.size())) {
	  if (spline_xsec_v[ebin] < 1e-5)
	    mcc8weight = 1e3;
	  else if (spline_xsec_v[ebin+binshift] < 1e-5)
	    mcc8weight = 0.;
	  else
	    mcc8weight = spline_xsec_v[ebin+binshift]/spline_xsec_v[ebin];
	}
	printf("for energy %f and energy bin %i the ratio is %f \n",nu_e,ebin,mcc8weight);
      }
      
      printf("\t found! \n");
      
      
      newtree->Fill();
      
   }// for all entries
   newtree->Print();
   newfile.Write();
}
