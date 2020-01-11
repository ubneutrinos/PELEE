#include "Riostream.h"

void slimmerTruth(TString fname)
{


  
   // Get old file, old tree and set top branch address
   TString dir = "/home/david/data/searchingfornues/v08_00_00_33/cc0pinp/0109/";
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
   for (auto activeBranchName : {"run","weights","weightSpline","weightTune","weightSplineTimesTune","nu_e","nu_pdg","leeweight" })
     oldtree->SetBranchStatus(activeBranchName, 1);
   
   int nu_pdg;
   oldtree->SetBranchAddress("nu_pdg", &nu_pdg);   
   
   // Create a new file + a clone of old tree in new file
   TFile newfile(foutname, "recreate");
   auto newtree = oldtree->CloneTree(0);
   for (auto i : ROOT::TSeqI(nentries)) {
      oldtree->GetEntry(i);

      // for numu files
      if (fabs(nu_pdg) == 12) continue;

      newtree->Fill();

   }// for all entries
   newtree->Print();
   newfile.Write();
}
