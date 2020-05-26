#include "Riostream.h"
// #include "TString.h"
// #include "TFile.h"
// #include "TTree.h"
#include <map>
#include <iostream>
#include <cstdlib>

void slimmer_1shr(TString finname)
{
  // Get old file, old tree and set top branch address
  TString foutname = "neutrinoselection_filt_1shr_skimmed.root";
  TFile oldfile(finname);
  TTree *oldtree;
  oldfile.GetObject("nuselection/NeutrinoSelectionFilter", oldtree);

  int numevts = 0;

  const auto nentries = oldtree->GetEntries();

  std::cout << "input file entries " << nentries << std::endl;

  for (auto b : *(oldtree->GetListOfBranches())) {
    if (std::strcmp(b->GetName(),"nslice") == 0)
      oldtree->SetBranchStatus(b->GetName(), 1);
    else if (std::strcmp(b->GetName(),"selected") == 0)
      oldtree->SetBranchStatus(b->GetName(), 1);
    else if (std::strcmp(b->GetName(),"run") == 0)
      oldtree->SetBranchStatus(b->GetName(), 1);
    else if (std::strcmp(b->GetName(),"sub") == 0)
      oldtree->SetBranchStatus(b->GetName(), 1);
    else if (std::strcmp(b->GetName(),"evt") == 0)
      oldtree->SetBranchStatus(b->GetName(), 1);
    else if (std::strcmp(b->GetName(),"n_showers_contained") == 0)
      oldtree->SetBranchStatus(b->GetName(), 1);
    else if (std::strcmp(b->GetName(),"n_tracks_contained") == 0)
      oldtree->SetBranchStatus(b->GetName(), 1);
    else if (std::strcmp(b->GetName(),"trk_llr_pid_score_v") == 0)
      oldtree->SetBranchStatus(b->GetName(), 1);
    else if (std::strcmp(b->GetName(),"trk_id") == 0)
      oldtree->SetBranchStatus(b->GetName(), 1);
    else if (std::strcmp(b->GetName(),"tksh_distance") == 0)
      oldtree->SetBranchStatus(b->GetName(), 1);
    else if (std::strcmp(b->GetName(),"shr_energy_tot_cali") == 0)
      oldtree->SetBranchStatus(b->GetName(), 1);
    else if (std::strcmp(b->GetName(),"category") == 0)
      oldtree->SetBranchStatus(b->GetName(), 1);
    else if (std::strcmp(b->GetName(),"CosmicIPAll3D") == 0)
      oldtree->SetBranchStatus(b->GetName(), 1);
    else if (std::strcmp(b->GetName(),"CosmicDirAll3D") == 0)
      oldtree->SetBranchStatus(b->GetName(), 1);
    else if (std::strcmp(b->GetName(),"_opfilter_pe_veto") == 0)
      oldtree->SetBranchStatus(b->GetName(), 1);
    else if (std::strcmp(b->GetName(),"_opfilter_pe_beam") == 0)
      oldtree->SetBranchStatus(b->GetName(), 1);
    else {
      std::cout << "setting branch " << b->GetName() << " to null" << std::endl;
      oldtree->SetBranchStatus(b->GetName(), 0);
    }
  }
  /*
  // Deactivate all branches
  //oldtree->SetBranchStatus("*", 0);
  oldtree->SetBranchStatus("nslice", 1);
  oldtree->SetBranchStatus("selected", 1);
  oldtree->SetBranchStatus("run", 1);
  oldtree->SetBranchStatus("sub", 1);
  oldtree->SetBranchStatus("evt", 1);
  oldtree->SetBranchStatus("n_showers_contained", 1);
  oldtree->SetBranchStatus("n_tracks_contained", 1);
  oldtree->SetBranchStatus("trk_llr_pid_score_v", 1);
  oldtree->SetBranchStatus("trk_id", 1);
  oldtree->SetBranchStatus("tksh_distance", 1);
  oldtree->SetBranchStatus("category", 1);
  //oldtree->SetBranchStatus("*", 0);
  */
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
  float tksh_distance;
  std::vector<float> *trk_llr_pid_score_v = new std::vector<float>();
  uint trk_id;
  

  oldtree->SetBranchAddress("nslice", &nslice);
  oldtree->SetBranchAddress("selected", &selected);
  oldtree->SetBranchAddress("trk_llr_pid_score_v", &trk_llr_pid_score_v);
  oldtree->SetBranchAddress("trk_id", &trk_id);
  oldtree->SetBranchAddress("tksh_distance", &tksh_distance);
  oldtree->SetBranchAddress("shr_energy_tot_cali", &shr_energy_tot_cali);
  oldtree->SetBranchAddress("n_tracks_contained", &n_tracks_contained);
  oldtree->SetBranchAddress("n_showers_contained", &n_showers_contained);
  oldtree->SetBranchAddress("n_tracks_contained", &n_tracks_contained);

			    
  // Create a new file + a clone of old tree in new file
  TFile newfile(foutname, "recreate");
  TDirectory *nuselection = newfile.mkdir("nuselection");
  nuselection->cd();

  auto newtree = oldtree->CloneTree(0);

  newtree->SetBranchStatus("shr_energy_tot_cali",0);

  std::cout << "Start loop with entries " << nentries << std::endl;

  for (auto i : ROOT::TSeqI(nentries))
  {
    if (i % 10000 == 0)
    {
      std::cout << "Entry num  " << i << std::endl;
    }

    oldtree->GetEntry(i);

    bool preseq = (nslice == 1) &&
         (selected == 1) &&
         (shr_energy_tot_cali > 0.07);

    //std::cout << "shower energy " << shr_energy_tot_cali << std::endl;

    bool np_preseq = preseq && (n_tracks_contained > 0) && (tksh_distance > 5.0);

    if (!np_preseq) continue;
    
    std::cout << "track PID vector has " << trk_llr_pid_score_v->size() << " elements" << std::endl;
    std::cout << "longest track at inedx " << trk_id  << std::endl;
    std::cout << "track PID is " << trk_llr_pid_score_v->at(trk_id-1) << std::endl;
    
    bool selection = np_preseq && (trk_llr_pid_score_v->at(trk_id-1) > 0.2);

    if (selection)
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
