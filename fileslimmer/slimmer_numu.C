#include "Riostream.h"
#include <map>

void slimmer_numu(TString fname,float splinexsecshift=0.)
{


  
   // Get old file, old tree and set top branch address
   TString dir = "/home/david/data/searchingfornues/v08_00_00_33/cc0pinp/0218/run3/";
   TString fullpath = dir + fname + ".root";
   TString textpath = dir + "txt/" + fname + "_numuconstraint.txt";
   TString foutname = dir + "SBNFit/" +  fname + "_numuconstraint" + ".root";
   gSystem->ExpandPathName(dir);
   //const auto filename = gSystem->AccessPathName(dir) ? "./Event.root" : "$ROOTSYS/test/Event.root";
   TFile oldfile(fullpath);
   TTree *oldtree;
   oldfile.GetObject("nuselection/NeutrinoSelectionFilter", oldtree);


   // load MCC8 SPLINE XSEC
   TString splinepath = "/home/david/Downloads/ccqe_spline_ratios.root";
   TFile splines(splinepath);
   TGraph *MCC9spline;
   splines.GetObject("nu_mu_ccqe_v304a", MCC9spline);
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
   //printf("spline bin width is %f \n",spline_binwidth);
   for (int n=0; n < spline_energy_v.size(); n++) {
     if (n%10 != 0) continue;
     float xsecratio = 1.;
     if ( (spline_energy_v[n] > 0.105) && (spline_energy_v[n+binshift] > 0.105) ) {
       if ( (n+binshift >= 0) && (n+binshift < spline_energy_v.size())) {
	 if (spline_xsec_v[n] < 1e-5)
	   xsecratio = 1e3;
	 else if (spline_xsec_v[n+binshift] < 1e-5)
	   xsecratio = 0.;
	 else
	   xsecratio = spline_xsec_v[n+binshift]/spline_xsec_v[n];
       }
     }
     //printf("the bin shift %f at energy %f is %f \n",splinexsecshift,spline_energy_v[n],xsecratio);
   }

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

  int numevts = 0;
  
  while (1) {
    if (!infile.good()) break;
    nlines += 1;
    infile >> runf >> subf >> evtf >> muonanglef >> muonenergyf >> neutrinoenergyf;
    if (run_event_map.find(runf) == run_event_map.end()) {

      std::vector<float> muoninfo{ muonanglef, muonenergyf, neutrinoenergyf };
      std::pair<int, std::vector<float> > eventinfo = std::make_pair( evtf, muoninfo );
      std::vector< std::pair<int, std::vector<float> > > evt_v{eventinfo};
      
      run_event_map[runf] = evt_v;

      numevts += 1;
      
    }
    else {

      std::vector<float> muoninfo{ muonanglef, muonenergyf, neutrinoenergyf };
      std::pair<int, std::vector<float> > eventinfo = std::make_pair( evtf, muoninfo );
            
      run_event_map[runf].push_back( eventinfo );

      numevts += 1;
      
    }
  }
  
  printf("there are %i events to be fetched! \n",numevts);
  
  const auto nentries = oldtree->GetEntries();

   // Deactivate all branches
   oldtree->SetBranchStatus("*", 0);
   // Activate only four of them
   for (auto activeBranchName : {"run","weights","nu_e","nslice","selected",
	 "nu_pdg","leeweight","weightSpline","weightTune","weightSplineTimesTune",
	 "run","sub","evt","npi0","category","ccnc","interaction"
	 })
      oldtree->SetBranchStatus(activeBranchName, 1);


   float weightSpline;
   int run,sub,evt;
   int ccnc;
   int interaction;
   int nslice;
   int nu_pdg;
   int selected;
   int crtveto;
   float nu_e;
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

   oldtree->SetBranchAddress("nu_e", &nu_e);
   
   //oldtree->SetBranchAddress("bdt_global", &bdt_global);
   oldtree->SetBranchAddress("nslice", &nslice);
   oldtree->SetBranchAddress("interaction", &interaction);
   oldtree->SetBranchAddress("ccnc", &ccnc);

   // new branch with weight = leeweight * weightSpline
   double eventweight;
   double reco_e;
   double muonangle, muonenergy, neutrinoenergy;
   double mcc8weight; // scaling for xsec shift
   
   // Create a new file + a clone of old tree in new file
   TFile newfile(foutname, "recreate");
   auto newtree = oldtree->CloneTree(0);
   newtree->Branch("eventweight",&eventweight,"eventweight/D");
   newtree->Branch("reco_e",&reco_e,"reco_e/D");
   newtree->Branch("muonangle",&muonangle,"muonangle/D");
   newtree->Branch("muonenergy",&muonenergy,"muonenergy/D");
   newtree->Branch("neutrinoenergy",&neutrinoenergy,"neutrinoenergy/D");
   newtree->Branch("mcc8weight",&mcc8weight,"mcc8weight/D");

   int nfill = 0;
   
   for (auto i : ROOT::TSeqI(nentries)) {

      oldtree->GetEntry(i);

      if (nslice == 1) { // &&  
	
	eventweight = weightSpline;
	reco_e = NeutrinoEnergy2/1000. + 0.105; // ((shr_energy_tot_cali+0.030)/0.79) + trk_energy_tot;

	if (i % 10000 == 0) 
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


	// set MCC8 weight
	mcc8weight = 1.0;
	if ( (interaction == 0) && (ccnc == 0) && (nu_e < 2.5) ) {
	  // energy bin for xsec ratio:
	  int ebin = int(nu_e/spline_binwidth);
	  if ( (spline_energy_v[ebin] > 0.105) && (spline_energy_v[ebin+binshift] > 0.105) ) {
	    if ( (ebin+binshift >= 0) && (ebin+binshift < spline_energy_v.size())) {
	      if (spline_xsec_v[ebin] < 1e-5)
		mcc8weight = 1e3;
	      else if (spline_xsec_v[ebin+binshift] < 1e-5)
		mcc8weight = 0.;
	      else
		mcc8weight = spline_xsec_v[ebin+binshift]/spline_xsec_v[ebin];
	    }
	  }
	  //printf("for energy %f and energy bin %i the ratio is %f \n",nu_e,ebin,mcc8weight);
	}

	
	//printf("\t found! \n");

	nfill += 1;
	
	newtree->Fill();
	
	
      }// if cuts pass
      
   }// for all entries

   printf("saved %i entries \n",nfill);
   
   newtree->Print();
   newfile.Write();
}
