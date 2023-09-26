//#define DEBUG

#include "TTree.h"
#include "TFile.h"

// Script to find the set of runs and subruns shared accross a set of PeLEE ntuples
// and create filtered versions only containig those runs and subruns, needed for
// detector systematic calculations
// Author: C Thorpe
//
// Useage root -b CheckRSLists.C
// Change the contents of the vector of files below to set which files it checks

const std::vector<std::string> files = {
"prodgenie_bnb_nu_overlay_DetVar_CV_reco2_v08_00_00_38_run3b_reco2_reco2.root",
"prodgenie_bnb_nu_overlay_DetVar_LYAttenuation_v08_00_00_38_run3b_reco2_reco2.root",
"prodgenie_bnb_nu_overlay_DetVar_LYDown_v08_00_00_37_v2_run3b_reco2_reco2.root",
"prodgenie_bnb_nu_overlay_DetVar_LYRayleigh_v08_00_00_37_run3b_reco2_reco2.root",
"prodgenie_bnb_nu_overlay_DetVar_Recomb2_reco2_v08_00_00_39_run3b_reco2_reco2.root",
"prodgenie_bnb_nu_overlay_DetVar_SCE_reco2_v08_00_00_38_run3b_reco2_reco2.root",
"prodgenie_bnb_nu_overlay_DetVar_WireModAngleXZ_v08_00_00_38_exe_run3b_reco2_reco2.root",
"prodgenie_bnb_nu_overlay_DetVar_WireModAngleYZ_v08_00_00_38_exe_run3b_reco2_reco2.root",
"prodgenie_bnb_nu_overlay_DetVar_wiremod_ScaledEdX_v08_00_00_39_run3b_reco2_reco2.root",
"prodgenie_bnb_nu_overlay_DetVar_wiremod_ScaleX_v08_00_00_38_run3b_reco2_reco2.root",
"prodgenie_bnb_nu_overlay_DetVar_wiremod_ScaleYZ_v08_00_00_38_run3b_reco2_reco2.root"
};

void CheckRSLists(){

  if(files.size() < 2){
    std::cout << "Input vector only has one file, you need at least 2 files" << std::endl;
    return;
  }

  gSystem->Exec("mkdir -p CommonRS");

  // iterate through the files, get the subrun tree from each

  std::vector<std::map<int,std::vector<int>>> rs_map(files.size()); // vector of maps, one map per file

  for(size_t i_f=0;i_f<files.size();i_f++){

    TFile* f_in = TFile::Open(files.at(i_f).c_str());
    TTree* sub_tree = static_cast<TTree*>(f_in->Get("nuselection/SubRun"));
    Int_t run,subRun;
    sub_tree->SetBranchAddress("run",&run); 
    sub_tree->SetBranchAddress("subRun",&subRun); 

    for(int ientry=0;ientry<sub_tree->GetEntries();ientry++){
      sub_tree->GetEntry(ientry);
      if(rs_map.at(i_f).find(run) == rs_map.at(i_f).end()) rs_map.at(i_f)[run] = {subRun};
      else rs_map.at(i_f)[run].push_back(subRun);
    }        

    delete sub_tree;
    f_in->Close();
  }

  // with the rs list, check which rs's are present in all of the files
  std::map<int,std::vector<int>> rs_map_shared;
  std::map<int,std::vector<int>>::iterator it_rs_map;

  // check everything against the 1st file in the list
  for(it_rs_map = rs_map.at(0).begin();it_rs_map != rs_map.at(0).end();it_rs_map++){

    bool found_run = true;
    const int* run = &it_rs_map->first;

    // first check the run exists
    for(size_t i_f=1;i_f<files.size();i_f++){
      if(rs_map.at(i_f).find(*run) == rs_map.at(i_f).end()){
        found_run = false;
        break;
      }
    }
    if(!found_run) continue; // don't need to check the subruns if the run is missing!

    // add the run (with no subruns) to the vector
    rs_map_shared[*run] = {};

    const std::vector<int>* sub = &it_rs_map->second; // pointer to the list of subruns at this run

    for(size_t i_s=0;i_s<sub->size();i_s++){
      bool missing_sub = false;
      for(size_t i_f=1;i_f<files.size();i_f++){
        if(std::find(rs_map.at(i_f)[*run].begin(),rs_map.at(i_f)[*run].end(),sub->at(i_s)) == rs_map.at(i_f)[*run].end()){
          missing_sub = true;
          break;
        }
      } // i_f

      if(!missing_sub) rs_map_shared[*run].push_back(sub->at(i_s));        

    } // loop over subruns in current run

  } // loop over runs in the 1st file

  // make a const copy of the rs_map_shared to be safe
  const std::map<int,std::vector<int>> c_rs_map_shared = rs_map_shared;
  rs_map_shared.clear(); // just to free up the memory

  std::cout << "Found " <<  c_rs_map_shared.size() << " runs that appear in all files,";
  std::cout << "the first file in the list you gave me contained " << rs_map.at(0).size() << " runs"  << std::endl;

  // Record how many events are in each file (should be identical) - issue a warning if they don't match
  int events_per_file;  
  int subruns_per_file;

  // brute-forcey debug - map between rs number and the events
  #ifdef DEBUG
  std::map<std::pair<int,int>,std::vector<int>> debug_rse_map; 
  #endif

  // Make a cloned version of the ntuple with only the common run/subruns
  for(size_t i_f=0;i_f<files.size();i_f++){

    std::cout << "Filtering " << files.at(i_f) << std::endl;

    TFile* f_in = TFile::Open(files.at(i_f).c_str());
    TTree* evt_tree = static_cast<TTree*>(f_in->Get("nuselection/NeutrinoSelectionFilter"));
    TTree* sub_tree = static_cast<TTree*>(f_in->Get("nuselection/SubRun"));

    // Create a new file + a clone of old tree in new file
    TFile* f_out = TFile::Open(("CommonRS/" + files.at(i_f)).c_str(),"recreate");
    f_out->mkdir("nuselection");
    f_out->cd("nuselection");
    TTree* new_evt_tree = evt_tree->CloneTree(0);
    Int_t run,sub,evt;
    evt_tree->SetBranchAddress("run",&run); 
    evt_tree->SetBranchAddress("sub",&sub); 
    evt_tree->SetBranchAddress("evt",&evt); 

    for(int ientry=0;ientry<evt_tree->GetEntries();ientry++){
      evt_tree->GetEntry(ientry);

      // for debugging only
      #ifdef DEBUG
      if(rs_map_shared.find(run) != rs_map_shared.end()){
        if(std::find(c_rs_map_shared.at(run).begin(),c_rs_map_shared.at(run).end(),sub) != c_rs_map_shared.at(run).end()){
          if(i_f == 0){
            if(debug_rse_map.find(std::make_pair(run,sub)) == debug_rse_map.end()) debug_rse_map[std::make_pair(run,sub)] = {evt};
            else debug_rse_map[std::make_pair(run,sub)].push_back(evt);
          }
          else{
            // if the event is missing from the 1st file
            if(std::find(debug_rse_map[std::make_pair(run,sub)].begin(),debug_rse_map[std::make_pair(run,sub)].end(),evt) == debug_rse_map[std::make_pair(run,sub)].end()){
              std::cout << "This event is missing from the 1st file " << run << " " << sub << " " << evt << std::endl;
            }
          }
        }
      }
      #endif

      if(c_rs_map_shared.find(run) != c_rs_map_shared.end()){
        if(std::find(c_rs_map_shared.at(run).begin(),c_rs_map_shared.at(run).end(),sub) != c_rs_map_shared.at(run).end())
          new_evt_tree->Fill(); 
      }

    }

    std::cout << "Found " << new_evt_tree->GetEntries() << " events" << std::endl;

    // Check the number of events in this file matches those in the previous files
    if(i_f == 0) events_per_file = new_evt_tree->GetEntries();
    else if(events_per_file != new_evt_tree->GetEntries()){
      std::cout << "WARNING: THE NUMBER OF EVENTS IN THIS FILE DOES NOT MATCH THE OTHERS" << std::endl;
      //throw std::invalid_argument("WARNING: THE NUMBER OF EVENTS IN THIS FILE DOES NOT MATCH THE OTHERS");
    }

    new_evt_tree->Write("NeutrinoSelectionFilter");

    // Do the same with the run/subrun/pot tree
    TTree* new_sub_tree = sub_tree->CloneTree(0);
    sub_tree->SetBranchAddress("run",&run); 
    sub_tree->SetBranchAddress("subRun",&sub); 

    std::map<int,std::vector<int>> rs_found;

    int fill_calls = 0;
    for(int ientry=0;ientry<sub_tree->GetEntries();ientry++){
      //std::cout << ientry << std::endl;
      sub_tree->GetEntry(ientry);

      bool found_run = c_rs_map_shared.find(run) != c_rs_map_shared.end();
      bool found_sub = found_run && std::find(c_rs_map_shared.at(run).begin(),c_rs_map_shared.at(run).end(),sub) != c_rs_map_shared.at(run).end();
      if(!found_sub) continue;

      new_sub_tree->Fill();

    }

    std::cout << "Found " << new_sub_tree->GetEntries() << " subruns" << std::endl;

    if(i_f == 0) subruns_per_file = new_sub_tree->GetEntries();
    else if(subruns_per_file != new_sub_tree->GetEntries()){
      std::cout << "WARNING: THE NUMBER OF SUBRUNS IN THIS FILE DOES NOT MATCH THE OTHERS" << std::endl;
      //throw std::invalid_argument("WARNING: THE NUMBER OF EVENTS IN THIS FILE DOES NOT MATCH THE OTHERS");
    }

    new_sub_tree->Write("SubRun");

    delete evt_tree;
    delete sub_tree;
    f_in->Close();
    f_out->Close();
  }

}
