#include "Riostream.h"
// #include "TString.h"
// #include "TFile.h"
// #include "TTree.h"
#include <map>
#include <iostream>
#include <cstdlib>

void make_dataset_1e0p_loose(TString finname)
{
  // Get file, tree and set top branch address

  TFile file(finname);
  TTree *tree;
  file.GetObject("nuselection/NeutrinoSelectionFilter", tree);

  int numevts = 0;

  const auto nentries = tree->GetEntries();

  std::cout << "input file entries " << nentries << std::endl;

  // Deactivate all branches
  tree->SetBranchStatus("*", 1);

  /*
    n_tracks_contained == 0 and n_showers_contained == 1 and CosmicIPAll3D > 10. and CosmicDirAll3D > -0.9 and CosmicDirAll3D < 0.9 and shrmoliereavg < 15 and subcluster > 4 and trkfit < 0.65 and secondshower_Y_nhit < 50
   */
  
  Int_t run;
  Int_t sub;

  Int_t nslice;
  Int_t selected;
  Float_t shr_energy_tot_cali;
  UInt_t n_tracks_contained;
  UInt_t n_showers_contained;
  Float_t reco_e;

  Float_t   CosmicIPAll3D;
  Float_t   CosmicDirAll3D;
  Float_t   hits_ratio;
  Float16_t shrmoliereavg;
  UInt_t    subcluster;
  Float_t   trkfit;
  Int_t     secondshower_Y_nhit;
  Float_t   bdt_bkg_0p;

  tree->SetBranchAddress("run", &run);
  tree->SetBranchAddress("sub", &sub);

  tree->SetBranchAddress("nslice", &nslice);
  tree->SetBranchAddress("selected", &selected);
  tree->SetBranchAddress("shr_energy_tot_cali", &shr_energy_tot_cali);
  tree->SetBranchAddress("n_tracks_contained", &n_tracks_contained);
  tree->SetBranchAddress("n_showers_contained", &n_showers_contained);
  tree->SetBranchAddress("reco_e", &reco_e);

  tree->SetBranchAddress("CosmicIPAll3D", &CosmicIPAll3D);
  tree->SetBranchAddress("CosmicDirAll3D", &CosmicDirAll3D);
  tree->SetBranchAddress("hits_ratio", &hits_ratio);
  tree->SetBranchAddress("shrmoliereavg", &shrmoliereavg);
  tree->SetBranchAddress("subcluster", &subcluster);
  tree->SetBranchAddress("trkfit", &trkfit);
  tree->SetBranchAddress("secondshower_Y_nhit", &secondshower_Y_nhit);

  tree->SetBranchAddress("bdt_bkg_0p", &bdt_bkg_0p);

  std::cout << "Start loop with entries " << nentries << std::endl;

  ofstream myfile;
  myfile.open("list_pelee_zp_lowe_loose.txt");
  
  for (auto i : ROOT::TSeqI(nentries))
  {
    if (i % 10000 == 0)
    {
      std::cout << "Entry num  " << i << std::endl;
    }

    tree->GetEntry(i);

    bool preseq = (nslice == 1) && (selected == 1) && (shr_energy_tot_cali > 0.07);

    bool zp_preseq = preseq && (n_tracks_contained == 0);

    bool zp_preseq_one_shower = zp_preseq && (n_showers_contained == 1);

    bool loose = ((n_tracks_contained == 0) && (n_showers_contained == 1) && (CosmicIPAll3D > 10.) && (CosmicDirAll3D > -0.9) && (CosmicDirAll3D < 0.9) && (shrmoliereavg < 15) && (subcluster > 4) && (trkfit < 0.65) && (secondshower_Y_nhit < 50));

    bool low_energy = (reco_e > 0.15)&&(reco_e < 0.65);

    bool pass_loose = zp_preseq_one_shower && low_energy && loose && (bdt_bkg_0p>0.4);

    if (pass_loose)
    {
      myfile << run << "." << sub << "\n";
    }// if cuts pass
  }// for all entries

  myfile.close();
  file.Close();
}
