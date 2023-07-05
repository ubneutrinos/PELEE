#include "Riostream.h"
#include <map>


void slimmer(TString path, TString runstr, TString fname, TString samplename, TString rootfolder)
{


  
   // Get old file, old tree and set top branch address
   TString dir = path + runstr + "/";
   TString fullpath = dir + fname + ".root";
   TString textpath = path + "fakedata/" + samplename + ".txt";
   TString foutname = dir + "SBNFit/" + fname + "_fakedata1_1eNp_BDT_sbnfit" + ".root";
   gSystem->ExpandPathName(dir);
   //const auto filename = gSystem->AccessPathName(dir) ? "./Event.root" : "$ROOTSYS/test/Event.root";
   TFile oldfile(fullpath);
   TTree *oldtree;
   //oldfile.GetObject("nuselection/NeutrinoSelectionFilter", oldtree);
   oldfile.GetObject(rootfolder+"/NeutrinoSelectionFilter", oldtree);

   printf("run : %s sample-name : %s file-name : %s \n",runstr.Data(),samplename.Data(),fname.Data());
   
   // load input text file with event/subrun/run
   ifstream infile;
   infile.open(textpath);
   
   int runf,subf,evtf;
   
   std::map<int,std::vector<int>> run_event_map, run_subrun_map;
   
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
       std::vector<int> sub_v = {subf};
       run_subrun_map[runf] = sub_v;
     }
     else {
       run_event_map[runf].push_back( evtf );
       run_subrun_map[runf].push_back( subf );
     }
   }
   
   printf("there are %i lines \n",nlines);
   
   const auto nentries = oldtree->GetEntries();
   
   // Deactivate all branches
   oldtree->SetBranchStatus("*", 0);
   // Activate only four of them
   for (auto activeBranchName : {"run","weights","shr_energy_tot","slpdg","nu_e","nslice","selected","NeutrinoEnergy2","crtveto","crthitpe","trk_len",
	 "topological_score","nu_pdg","leeweight","weightSpline","weightTune","weightSplineTimesTune",
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
   oldtree->SetBranchAddress("sub", &sub);
   oldtree->SetBranchAddress("evt", &evt);
   oldtree->SetBranchAddress("interaction", &interaction);
   oldtree->SetBranchAddress("ccnc",&ccnc);
   oldtree->SetBranchAddress("nu_e",&nu_e);
   //oldtree->SetBranchAddress("category",&category);
   
   oldtree->SetBranchAddress("trk_len",&trk_len);
   oldtree->SetBranchAddress("NeutrinoEnergy2",&NeutrinoEnergy2);
   oldtree->SetBranchAddress("shr_energy_tot_cali",&shr_energy_tot_cali);
   oldtree->SetBranchAddress("trk_energy_tot",&trk_energy_tot);
   
   // new branch with weight = leeweight * weightSpline
   float eventweight;
   float reco_e;
   double truenuenergy;
   
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
   newtree->Branch("truenuenergy",&truenuenergy,"truenuenergy/D");
   
   for (auto i : ROOT::TSeqI(nentries)) {
     oldtree->GetEntry(i);

     //eventweight = leeweight * weightSpline;
     reco_e = ((shr_energy_tot_cali+0.0)/0.83) + trk_energy_tot;
     
     
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
     auto sub_v = run_subrun_map[run];
     
     bool found = false;
     for (size_t ne=0; ne < evt_v.size(); ne++) {
       if ( (evt == evt_v[ne]) && (sub == sub_v[ne]) ) {
	 found = true;
	 break;
       }
     }
     
     if (found == false) continue;
     
     truenuenergy = nu_e;
     
     
     newtree->Fill();
     
   }// for all entries
   //newtree->Print();
   printf("new tree saved with %lli entries \n",newtree->GetEntries());
   newfile.Write();
}


void slimmerFakeData(TString path) {
  
  TString run1 = "run1";
  TString run2 = "run2";
  TString run3 = "run3";

  TString rootfolder = "searchingfornues";

  std::vector<TString> string_v, sample_v;

  TString s1 = "mc";
  TString f1 = "prodgenie_bnb_nu_uboone_overlay_mcc9.1_v08_00_00_26_filter_run1_reco2_reco2_nuepresel";
  TString s2 = "ncpi0";
  TString f2 = "prodgenie_nc_pi0_uboone_overlay-v08_00_00_26_run1_reco2_reco2_nuepresel";
  TString s3 = "ccpi0";
  TString f3 = "prodgenie_cc_pi0_uboone_overlay_v08_00_00_26_run1_reco2_nuepresel";
  TString s4 = "ccnopi";
  TString f4 = "prodgenie_CCmuNoPi_overlay_mcc9_v08_00_00_33_all_run1_reco2_reco2_nuepresel";
  TString s5 = "ncnopi";
  TString f5 = "prodgenie_ncnopi_overlay_mcc9_v08_00_00_33_run1_reco2_reco2_nuepresel";
  TString s6 = "nccpi";
  TString f6 = "prodgenie_NCcPiNoPi0_overlay_mcc9_v08_00_00_33_run1_reco2_reco2_nuepresel";
  TString s7 = "nue";
  TString f7 = "prodgenie_bnb_intrinsice_nue_uboone_overlay_mcc9.1_v08_00_00_26_run1_reco2_reco2_nuepresel";
  TString s8 = "cccpi";
  TString f8 = "prodgenie_filter_CCmuCPiNoPi0_overlay_mcc9_v08_00_00_33_run1_reco2_reco2_nuepresel";
  
  string_v = {f1,f2,f3,f4,f5,f6,f7,f8};
  sample_v = {s1,s2,s3,s4,s5,s6,s7,s8};
  for (size_t i=0; i < string_v.size(); i++)
    slimmer(path,run1,string_v[i],sample_v[i],rootfolder);

  TString s = "data";
  TString f = "prod_uboone_nu2020_fakedata_set1_run1_reco2_v08_00_00_41_reco2";
  slimmer(path,"fakedata",f,s,"nuselection");
  
  f1 = "prodgenie_bnb_nu_uboone_overlay_mcc9.1_v08_00_00_26_filter_run3_reco2_G_reco2_nuepresel";
  f2 = "prodgenie_nc_pi0_uboone_overlay_mcc9.1_v08_00_00_26_run3_G_reco2_nuepresel";
  f3 = "prodgenie_cc_pi0_uboone_overlay_v08_00_00_26_run3_G_reco2_nuepresel";
  f4 = "prodgenie_CCmuNoPi_overlay_mcc9_v08_00_00_33_all_run3_reco2_reco2_nuepresel";
  f5 = "prodgenie_ncnopi_overlay_mcc9_v08_00_00_33_new_run3_reco2_reco2_nuepresel";
  f6 = "prodgenie_NCcPiNoPi0_overlay_mcc9_v08_00_00_33_New_run3_reco2_reco2_nuepresel";
  f7 = "prodgenie_bnb_intrinsice_nue_uboone_overlay_mcc9.1_v08_00_00_26_run3_reco2_reco2_nuepresel";
  f8 = "prodgenie_filter_CCmuCPiNoPi0_overlay_mcc9_v08_00_00_33_run3_reco2_reco2_nuepresel";
  
  string_v = {f1,f2,f3,f4,f5,f6,f7,f8};
  for (size_t i=0; i < string_v.size(); i++)
    slimmer(path,run3,string_v[i],sample_v[i],rootfolder);

  s = "data";
  f = "prod_uboone_nu2020_fakedata_set1_run3b_reco2_v08_00_00_41_reco2";
  slimmer(path,"fakedata",f,s,"nuselection");

  s1 = "mc";
  f1 = "prodgenie_bnb_nu_uboone_overlay_mcc9.1_v08_00_00_26_filter_run2_reco2_D1D2_reco2_nuepresel";
  s2 = "nue";
  f2 = "prodgenie_bnb_intrinsic_nue_overlay_run2_v08_00_00_35_run2a_reco2_reco2_nuepresel";
  string_v = {f1,f2};
  sample_v = {s1,s2};
  for (size_t i=0; i < string_v.size(); i++)
    slimmer(path,run2,string_v[i],sample_v[i],rootfolder);
  
  return;
  
}
