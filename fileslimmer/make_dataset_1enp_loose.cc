#include "Riostream.h"
// #include "TString.h"
// #include "TFile.h"
// #include "TTree.h"
#include <map>
#include <iostream>
#include <cstdlib>

void make_dataset_1enp_loose(TString finname)
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
    nslice == 1 and selected == 1 and shr_energy_tot_cali > 0.07 and n_tracks_contained > 0 and CosmicIPAll3D > 10. and trkpid < 0.02 and hits_ratio > 0.50 and shrmoliereavg < 9 and subcluster > 4 and trkfit < 0.65 and tksh_distance < 6.0 and shr_tkfit_dedx_max > 0.5 and shr_tkfit_dedx_max < 5.5 and tksh_angle > -0.9 and n_showers_contained == 1
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
  Float_t   trkpid;
  Float_t   hits_ratio;
  Float16_t shrmoliereavg;
  UInt_t    subcluster;
  Float_t   trkfit;
  Float_t   tksh_distance;
  Float_t   shr_tkfit_dedx_max;
  Float_t   tksh_angle;
  
  tree->SetBranchAddress("run", &run);
  tree->SetBranchAddress("sub", &sub);

  tree->SetBranchAddress("nslice", &nslice);
  tree->SetBranchAddress("selected", &selected);
  tree->SetBranchAddress("shr_energy_tot_cali", &shr_energy_tot_cali);
  tree->SetBranchAddress("n_tracks_contained", &n_tracks_contained);
  tree->SetBranchAddress("n_showers_contained", &n_showers_contained);
  tree->SetBranchAddress("reco_e", &reco_e);

  tree->SetBranchAddress("CosmicIPAll3D", &CosmicIPAll3D);
  tree->SetBranchAddress("trkpid", &trkpid);
  tree->SetBranchAddress("hits_ratio", &hits_ratio);
  tree->SetBranchAddress("shrmoliereavg", &shrmoliereavg);
  tree->SetBranchAddress("subcluster", &subcluster);
  tree->SetBranchAddress("trkfit", &trkfit);
  tree->SetBranchAddress("tksh_distance", &tksh_distance);
  tree->SetBranchAddress("shr_tkfit_dedx_max", &shr_tkfit_dedx_max);
  tree->SetBranchAddress("tksh_angle", &tksh_angle);

  std::cout << "Start loop with entries " << nentries << std::endl;

  ofstream myfile;
  myfile.open("list_pelee_np_lowe_loose.txt");
  
  for (auto i : ROOT::TSeqI(nentries))
  {
    if (i % 10000 == 0)
    {
      std::cout << "Entry num  " << i << std::endl;
    }

    tree->GetEntry(i);

    bool preseq = (nslice == 1) && (selected == 1) && (shr_energy_tot_cali > 0.07);

    bool np_preseq = preseq && (n_tracks_contained > 0);

    bool np_preseq_one_shower = np_preseq && (n_showers_contained == 1);

    bool loose = ((CosmicIPAll3D > 10.) && (trkpid < 0.02) && (hits_ratio > 0.50) && (shrmoliereavg < 9) && (subcluster > 4) && (trkfit < 0.65) && (tksh_distance < 6.0) && (shr_tkfit_dedx_max > 0.5) && (shr_tkfit_dedx_max < 5.5) && (tksh_angle > -0.9) && (n_showers_contained == 1));

    bool low_energy = (reco_e > 0.15)&&(reco_e < 0.65);

    bool pass_loose = np_preseq_one_shower && low_energy && loose;

    if (pass_loose)
    {
      myfile << run << "." << sub << "\n";
    }// if cuts pass
  }// for all entries

  myfile.close();
  file.Close();
}
