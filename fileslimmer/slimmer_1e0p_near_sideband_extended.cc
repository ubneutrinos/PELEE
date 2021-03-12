#include "Riostream.h"
// #include "TString.h"
// #include "TFile.h"
// #include "TTree.h"
#include <map>
#include <iostream>
#include <cstdlib>

void slimmer_1e0p_far_sideband_extended(TString finname)
{
  // Get old file, old tree and set top branch address
  //TString finname = "/uboone/data/users/cerati/searchingfornues/v08_00_00_42/cc0pinp/0414/run1/data_bnb_peleeTuple_unblinded_uboone_v08_00_00_42_run1_C1_nuepresel.root";
  TString foutname = "neutrinoselection_filt_1e0p_far_sideband_skimmed.root";
  TFile oldfile(finname);
  TTree *oldtree;
  oldfile.GetObject("nuselection/NeutrinoSelectionFilter", oldtree);

  int numevts = 0;

  const auto nentries = oldtree->GetEntries();

  // Deactivate all branches
  oldtree->SetBranchStatus("*", 1);

  int backtracked_pdg;

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

  oldtree->SetBranchAddress("nslice", &nslice);
  oldtree->SetBranchAddress("selected", &selected);
  oldtree->SetBranchAddress("contained_fraction", &contained_fraction);
  oldtree->SetBranchAddress("shr_energy_tot_cali", &shr_energy_tot_cali);
  oldtree->SetBranchAddress("n_tracks_contained", &n_tracks_contained);
  oldtree->SetBranchAddress("n_showers_contained", &n_showers_contained);
  oldtree->SetBranchAddress("bdt_bkg_0p", &bdt_bkg_0p);
  oldtree->SetBranchAddress("reco_e", &reco_e);
  oldtree->SetBranchAddress("n_showers", &n_showers);

  // Create a new file + a clone of old tree in new file
  TFile newfile(foutname, "recreate");
  TDirectory *nuselection = newfile.mkdir("nuselection");
  nuselection->cd();

  auto newtree = oldtree->CloneTree(0);

  std::cout << "Start loop with entries " << nentries << std::endl;

  for (auto i : ROOT::TSeqI(nentries))
  {
    if (i % 10000 == 0)
    {
      std::cout << "Entry num  " << i << std::endl;
    }

    oldtree->GetEntry(i);

    //bool preseq = (nslice == 1) && (selected == 1) && (shr_energy_tot_cali > 0.07) && (contained_fraction > 0.9) && (n_showers_contained > 0) && (n_tracks_contained == 0);
    bool preseq = (nslice == 1) && (shr_energy_tot_cali > 0.07) && (contained_fraction > 0.4) && (n_showers_contained > 0) && (n_tracks_contained == 0);
    bool presel1e0p = preseq && (n_showers_contained == 1);

    bool low_bkg_score = bdt_bkg_0p < 0.72;
    bool high_energy = reco_e > 0.65;
    bool far_sideband = presel1e0p && (low_bkg_score || high_energy);

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

    if (far_sideband)
    {
      std::cout << "saving  " << i << std::endl;
      newtree->Fill();
    }// if cuts pass
  }// for all entries
  newtree->Print();

  TTree *subrunTree;
  oldfile.GetObject("nuselection/SubRun", subrunTree);
  auto newSubrunTree = subrunTree->CloneTree();
  newSubrunTree->Print();

  newfile.Write();
}
