#include <iostream>
#include <sys/stat.h>
#include <fstream>
#include <vector>

#include "TFile.h"
#include "TTree.h"
#include "TEventList.h"
#include "TMath.h"
#include "TROOT.h"
#include "TApplication.h"
#include "TH1.h"
#include "TH2.h"
#include "TCanvas.h"
#include "TF1.h"
#include "TColor.h"
#include "TGraph.h"
#include "TStyle.h"
#include "TGaxis.h"
#include "TSpectrum.h"
#include "TPolyMarker.h"
#include "TTreeReader.h"
#include "TParticle.h"
#include "TTreeReaderArray.h"

#include "include/padHelper.hh"

void analyze(TString path, Int_t max_iter){

    // +-----------+
    // | load file |
    // +-----------+
    auto *f = new TFile(path.Data());

    TTreeReader reader("g4hyptpc", f);
    TTreeReaderValue<Int_t> evnum(reader, "evnum");
    TTreeReaderValue<std::vector<TParticle>> TPC(reader,  "TPC");

    // +------------------------------------+
    // | event selection and write csv file |
    // +------------------------------------+
    // -- prepare --------------------------------
    TString save_name;
    Int_t dot_index = path.Last('.');
    Int_t sla_index = path.Last('/');
    for (Int_t i = sla_index+1; i < dot_index; i++) save_name += path[i];
    system(Form("rm ./csv_data/%s.csv", save_name.Data()));
    std::ofstream ofs(Form("./csv_data/%s.csv", save_name.Data()), std::ios::app);
    ofs << "evnum,x,y,z,pad_id,mom\n";

    // -- event selection and write --------------------------------
    Int_t pad_id = 0;
    TVector3 mom;
    reader.Restart();
    while (reader.Next() ){
        if (*evnum > max_iter) break;
        std::vector<Double_t> x;
        std::vector<Double_t> z;
        for(const auto& p_tpc : (*TPC)) {
            x.push_back(p_tpc.Vx());
            z.push_back(p_tpc.Vz());
            mom.SetXYZ( p_tpc.Px(), p_tpc.Py(), p_tpc.Pz() );
            pad_id = padHelper::findPadID(p_tpc.Vz(), p_tpc.Vx());
            // std::cout << *evnum << "," << p_tpc.Vx() << "," << p_tpc.Vy() << "," << p_tpc.Vz() << "," << pad_id << std::endl;
            if (0 <= pad_id && pad_id <= 5768) ofs << *evnum << "," << p_tpc.Vx() << "," << p_tpc.Vy() << "," << p_tpc.Vz() << "," << pad_id << "," << mom.Mag() << "\n";
        }
    }

    ofs.close();
    std::cout << "finish" << std::endl;
}

Int_t main(int argc, char** argv) { 
    TString path = argv[1];
    Int_t max_iter = 500000;
    if (argc > 2) max_iter = atoi(argv[2]);
    analyze(path, max_iter);
    return 0;
}