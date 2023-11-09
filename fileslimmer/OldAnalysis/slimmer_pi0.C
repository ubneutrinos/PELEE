#include "Riostream.h"
#include <map>

void slimmer_nuepresel(TString fname)
{


  
   // Get old file, old tree and set top branch address
   TString dir = "/home/david/data/searchingfornues/v08_00_00_33/cc0pinp/0218/run3/";
   TString fullpath = dir + fname + ".root";
   TString foutname = dir + fname + "_pi0sel" + ".root";
   gSystem->ExpandPathName(dir);
   //const auto filename = gSystem->AccessPathName(dir) ? "./Event.root" : "$ROOTSYS/test/Event.root";
   TFile oldfile(fullpath);
   TTree *oldtree;
   oldfile.GetObject("nuselection/NeutrinoSelectionFilter", oldtree);


  int numevts = 0;
  
  const auto nentries = oldtree->GetEntries();

   // Deactivate all branches
   oldtree->SetBranchStatus("*", 1);

   int nslice;
   int selected;
   int filter_pi0;
   float shr_energy_tot_cali;

   oldtree->SetBranchAddress("nslice", &nslice);
   oldtree->SetBranchAddress("selected", &selected);
   oldtree->SetBranchAddress("filter_pi0", &filter_pi0);
   oldtree->SetBranchAddress("shr_energy_tot_cali", &shr_energy_tot_cali);
   
   // Create a new file + a clone of old tree in new file
   TFile newfile(foutname, "recreate");
   TDirectory *searchingfornues = newfile.mkdir("searchingfornues");
   searchingfornues->cd();
   
   auto newtree = oldtree->CloneTree(0);

   for (auto i : ROOT::TSeqI(nentries)) {

      oldtree->GetEntry(i);

      if (filter_pi0 == 1) {
	
	newtree->Fill();
	
	
      }// if cuts pass
      
   }// for all entries
   newtree->Print();
   newfile.Write();
}
