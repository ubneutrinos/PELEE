#include "Riostream.h"
// #include "TString.h"
// #include "TFile.h"
// #include "TTree.h"
#include <map>
#include <iostream>
#include <cstdlib>

void slimmer_1enp_far_sideband(TString finname)
{
  // Get old file, old tree and set top branch address
  //TString finname = "/home/nic/Desktop/MicroBooNE/bnb_nue_analysis/PELEE/0304/run1/nuepresel/data_bnb_mcc9.1_v08_00_00_25_reco2_C1_beam_good_reco2_5e19_nuepresel.root";
  TString foutname = "neutrinoselection_filt_1enp_far_sideband_skimmed.root";
  TFile oldfile(finname);
  TTree *oldtree;
  oldfile.GetObject("nuselection/NeutrinoSelectionFilter", oldtree);

  int numevts = 0;

  const auto nentries = oldtree->GetEntries();

  std::cout << "input file entries " << nentries << std::endl;

  // Deactivate all branches
  oldtree->SetBranchStatus("*", 1);

  int backtracked_pdg;

  int nslice;
  int selected;
  float shr_energy_tot_cali;
  float _opfilter_pe_beam;
  float _opfilter_pe_veto;
  int bnbdata;
  int extdata;
  uint n_tracks_contained;
  uint n_showers_contained;
  float bdt_pi0_np;
  float bdt_nonpi0_np;
  float reco_e;
  int n_showers;
  float contained_fraction;

  oldtree->SetBranchAddress("nslice", &nslice);
  oldtree->SetBranchAddress("selected", &selected);
  oldtree->SetBranchAddress("shr_energy_tot_cali", &shr_energy_tot_cali);
  oldtree->SetBranchAddress("n_tracks_contained", &n_tracks_contained);
  oldtree->SetBranchAddress("n_showers_contained", &n_showers_contained);
  oldtree->SetBranchAddress("bdt_pi0_np", &bdt_pi0_np);
  oldtree->SetBranchAddress("bdt_nonpi0_np", &bdt_nonpi0_np);
  oldtree->SetBranchAddress("reco_e", &reco_e);
  oldtree->SetBranchAddress("n_showers", &n_showers);
  oldtree->SetBranchAddress("contained_fraction", &contained_fraction);


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

    bool preseq = (nslice == 1) &&
      //(selected == 1) &&
      ( ( (contained_fraction > 0.4) && (n_showers > 0) ) || (selected==1)) &&
      (shr_energy_tot_cali > 0.07);

    /*
    if (preseq)
      std::cout << "preseq++" << std::endl;
    else
      continue;
    */

    bool np_preseq = preseq && (n_tracks_contained > 0);

    /*
    if (np_preseq)
      std::cout << "np_preseq++" << std::endl;
    else
      continue;
    */

    bool np_preseq_one_shower = np_preseq && (n_showers_contained == 1);

    /*
    if (np_preseq_one_shower)
      std::cout << "np_preseq_one_shower++" << std::endl;
    else
      continue;
    */

    bool low_pid = ((bdt_pi0_np > 0) && (bdt_pi0_np < 0.1)) || ((bdt_nonpi0_np > 0) && (bdt_nonpi0_np < 0.1));

    /*
    if (low_pid)
      std::cout << "low_pid++" << std::endl;
    */

    bool high_energy = (reco_e > 0.85);

    /*
    if (high_energy)
      std::cout << "high_energy++" << std::endl;
    */

    bool far_sideband = np_preseq_one_shower && (low_pid || high_energy);

    /*
    if (far_sideband)
      std::cout << "far_sideband++" << std::endl;
    */

    if (far_sideband)
    {
      //std::cout << "far sideband pass!" << std::endl;
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
