#include "Riostream.h"
// #include "TString.h"
// #include "TFile.h"
// #include "TTree.h"
#include <map>
#include <iostream>
#include <cstdlib>

void tester_1e0p_unblinding(TString fSIDEBAND, TString fBLIND)
{
  
  // Get SIDEBAND file & tree and set top branch address
  TFile SIDEBANDfile(fSIDEBAND);
  TTree *SIDEBANDtree;
  SIDEBANDfile.GetObject("nuselection/NeutrinoSelectionFilter", SIDEBANDtree);
  
  // Get BLIND file & tree and set top branch address
  TFile BLINDfile(fBLIND);
  TTree *BLINDtree;
  BLINDfile.GetObject("nuselection/NeutrinoSelectionFilter", BLINDtree);
  
  
  const auto nentriesSIDEBAND = SIDEBANDtree->GetEntries();
  const auto nentriesBLIND    = BLINDtree->GetEntries();
  
  // Deactivate all branches
  SIDEBANDtree->SetBranchStatus("*", 1);
  BLINDtree->SetBranchStatus("*", 1);
  
  int backtracked_pdg;
  
  int run,sub,evt;
  int nslice;
  int selected;
  float shr_energy_tot_cali;
  float _opfilter_pe_beam;
  float _opfilter_pe_veto;
  uint n_tracks_contained;
  uint n_showers_contained;
  float bdt_bkg_0p;
  float reco_e;
  float contained_fraction;
  int n_showers;
  
  // maps linking run -> std::vector< event >
  std::map<int,std::vector<int>> SIDEBAND_PRESEL_MAP;
  std::map<int,std::vector<int>> SIDEBAND_FAR_MAP;
  
  SIDEBANDtree->SetBranchAddress("run", &run);
  SIDEBANDtree->SetBranchAddress("sub", &sub);
  SIDEBANDtree->SetBranchAddress("evt", &evt);
  SIDEBANDtree->SetBranchAddress("nslice", &nslice);
  SIDEBANDtree->SetBranchAddress("selected", &selected);
  SIDEBANDtree->SetBranchAddress("contained_fraction", &contained_fraction);
  SIDEBANDtree->SetBranchAddress("shr_energy_tot_cali", &shr_energy_tot_cali);
  SIDEBANDtree->SetBranchAddress("n_tracks_contained", &n_tracks_contained);
  SIDEBANDtree->SetBranchAddress("n_showers_contained", &n_showers_contained);
  SIDEBANDtree->SetBranchAddress("bdt_bkg_0p", &bdt_bkg_0p);
  SIDEBANDtree->SetBranchAddress("reco_e", &reco_e);
  SIDEBANDtree->SetBranchAddress("n_showers", &n_showers);
  
  std::cout << "Start loop with entries " << nentries << std::endl;
  
  for (auto i : ROOT::TSeqI(nentries)) {

    if (i % 10000 == 0)
      std::cout << "Entry num  " << i << std::endl;
    
    SIDEBANDtree->GetEntry(i);
    
    bool preseq = (nslice == 1) && (shr_energy_tot_cali > 0.07) && (contained_fraction > 0.4) && (n_showers_contained > 0) && (n_tracks_contained == 0);
    bool presel1e0p = preseq && (n_showers_contained == 1);
    
    bool low_bkg_score = bdt_bkg_0p < 0.4;
    bool high_energy = reco_e > 0.9;
    bool far_sideband = (low_bkg_score || high_energy);
    
    if (presel1e0p && far_sideband) {
      
      if (SIDEBAND_FAR_MAP.find(run) == SIDEBAND_FAR_MAP.end()) {
	std::vector<int> evt_v = {evt};
	SIDEBAND_FAR_MAP[run] = evt_v;
      }
      else {
	SIDEBAND_FAR_MAP[run].push_back( evt );
      }
    }

    /*
    if (preseq)
      std::cout << "presq ++ " << std::endl;
    if (presel1e0p)
      std::cout << "presel1e0p ++ " << std::endl;
    if (low_bkg_score)
      std::cout << "low_bkg_score ++ " << std::endl;
    if (high_energy)
      std::cout << "high_energy ++ " << std::endl;
    if (far_sideband)
      std::cout << "far_sideband ++ " << std::endl;
    */

    if (presel1e0p && !far_sideband)
    {
      std::cout << "you can't end up here! test failed." << std::endl;
      std::cout << "at entry i=" << i << " reco_e=" << reco_e << " bdt_bkg_0p=" << bdt_bkg_0p << std::endl;
      exit(1);
    }// if cuts pass
    
  }// for all entries of SIDEBAND tree
  
  
  // maps linking run -> std::vector< event >
  std::map<int,std::vector<int>> BLIND_PRESEL_MAP;
  std::map<int,std::vector<int>> BLIND_FAR_MAP;
  
  BLINDtree->SetBranchAddress("run", &run);
  BLINDtree->SetBranchAddress("sub", &sub);
  BLINDtree->SetBranchAddress("evt", &evt);
  BLINDtree->SetBranchAddress("nslice", &nslice);
  BLINDtree->SetBranchAddress("selected", &selected);
  BLINDtree->SetBranchAddress("contained_fraction", &contained_fraction);
  BLINDtree->SetBranchAddress("shr_energy_tot_cali", &shr_energy_tot_cali);
  BLINDtree->SetBranchAddress("n_tracks_contained", &n_tracks_contained);
  BLINDtree->SetBranchAddress("n_showers_contained", &n_showers_contained);
  BLINDtree->SetBranchAddress("bdt_bkg_0p", &bdt_bkg_0p);
  BLINDtree->SetBranchAddress("reco_e", &reco_e);
  BLINDtree->SetBranchAddress("n_showers", &n_showers);
  
  std::cout << "Start loop with entries " << nentries << std::endl;
  
  for (auto i : ROOT::TSeqI(nentries)) {

    if (i % 10000 == 0)
      std::cout << "Entry num  " << i << std::endl;
    
    BLINDtree->GetEntry(i);
    
    bool preseq = (nslice == 1) && (shr_energy_tot_cali > 0.07) && (contained_fraction > 0.4) && (n_showers_contained > 0) && (n_tracks_contained == 0);
    bool presel1e0p = preseq && (n_showers_contained == 1);
    
    bool low_bkg_score = bdt_bkg_0p < 0.4;
    bool high_energy = reco_e > 0.9;
    bool far_sideband = (low_bkg_score || high_energy);
    
    if (presel1e0p && far_sideband) {
      
      if (BLIND_FAR_MAP.find(run) == BLIND_FAR_MAP.end()) {
	std::vector<int> evt_v = {evt};
	BLIND_FAR_MAP[run] = evt_v;
      }
      else {
	BLIND_FAR_MAP[run].push_back( evt );
      }
    }

    /*
    if (preseq)
      std::cout << "presq ++ " << std::endl;
    if (presel1e0p)
      std::cout << "presel1e0p ++ " << std::endl;
    if (low_bkg_score)
      std::cout << "low_bkg_score ++ " << std::endl;
    if (high_energy)
      std::cout << "high_energy ++ " << std::endl;
    if (far_sideband)
      std::cout << "far_sideband ++ " << std::endl;
    */

    if (presel1e0p && !far_sideband)
    {
      std::cout << "you can't end up here! test failed." << std::endl;
      std::cout << "at entry i=" << i << " reco_e=" << reco_e << " bdt_bkg_0p=" << bdt_bkg_0p << std::endl;
      exit(1);
    }// if cuts pass
    
  }// for all entries of BLIND tree


  // are there the same number of entries in the far-sideband?
  if (far_sideband_BLIND != far_sideband_SIDEBAND) {
    std::cout << "FAIL -> did not find the sane number of far-sideband events";
    exit(1);
  }

  int nfail = 0;
  
  // now check that the SIDEBAND and BLIND files have the same entries
  for (itB = BLIND_FAR_MAP.begin(); itB != BLIND_FAR_MAP.end(); itB++) {

    int runBLIND = itB->first;
    std::vector<int> evtBLIND_v = itB->second;
    
    for (itS = SIDENBAND_FAR_MAP.begin(); itS != SIDEBAND_FAR_MAP.end(); itS++) {

      int runSIDEBAND = itS->first;
      std::vector<int> evtSIDEBAND_v = itS->second;

      if (runSIDEBAND != runBLIND) continue;

      // if made it this far the runs match, check event-by-event
      
      for (size_t iB=0; iB < evtBLIND_v.size(); iB++) {

	int evtBLIND = evtBLIND_v[iB];
	
	bool MATCH = false;
	
	for (size_t iS=0; iS < evtSIDEBAND_v.size(); iS++) {

	  int evtSIDEBAND = evtSIDEBAND_v[iS];

	  if (evtSIDEBAND == evtBLIND){
	    MATCH = true;
	    break;
	  }

	  // if not matched this is an error!
	  if (MATCH == false) {
	    std::cout << "FAIL -> [run,event] " << runBLIND << ", " << evtSIDEBAND << "] not fund in SIDEBAND file." << std::endl;
	    nfail += 1;
	  }

	}// for all blind-file events matched to this run
	
      }// for all sideband-file events matched to this run

    }// for all run vectors in SIDEBAND-file.
  }// for all vectors in BLIND-file.

  if (nfail == 0){
    std::cout << "FAIL -> not all events that pass the far-sideband cuts match one-to-one." << std::endl;
    exit(1);
  }

  std::cout << "test passed!" << std::endl;
}
