#include "Riostream.h"
#include <map>

void slimmer_nuepresel(TString fname)
{


  
   // Get old file, old tree and set top branch address
   TString dir = "/uboone/data/users/davidc/searchingfornues/v08_00_00_43/0702/run1/";
   TString fullpath = dir + fname + ".root";
   TString foutname = dir + fname + "_nuepresel" + ".root";
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
   uint n_showers_contained;
   float shr_energy_tot_cali;
   float contained_fraction;

   oldtree->SetBranchAddress("nslice", &nslice);
   oldtree->SetBranchAddress("selected", &selected);
   oldtree->SetBranchAddress("contained_fraction", &contained_fraction);
   oldtree->SetBranchAddress("shr_energy_tot_cali", &shr_energy_tot_cali);
   oldtree->SetBranchAddress("n_showers_contained", &n_showers_contained);
   
   // Create a new file + a clone of old tree in new file
   TFile newfile(foutname, "recreate");
   TDirectory *searchingfornues = newfile.mkdir("searchingfornues");
   searchingfornues->cd();
   
   auto newtree = oldtree->CloneTree(0);

   for (auto i : ROOT::TSeqI(nentries)) {

      oldtree->GetEntry(i);

      //if ( (nslice == 1) && (selected==1) && (shr_energy_tot_cali > 0.07) ) { // &&
      if ( (nslice == 1) && (contained_fraction > 0.4) && (n_showers_contained > 0) && (shr_energy_tot_cali > 0.07) ) {
	
	newtree->Fill();
	
	
      }// if cuts pass
      
   }// for all entries
   newtree->Print();
   newfile.Write();
}
