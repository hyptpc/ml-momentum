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

void analyze(TString path, Int_t event_id){

    // +-----------+
    // | load file |
    // +-----------+
    auto *f = new TFile(path.Data());

    TTreeReader reader("g4hyptpc", f);
    TTreeReaderValue<Int_t> evnum(reader, "evnum");
    TTreeReaderValue<std::vector<TParticle>> TPC(reader,  "TPC");

    // +----------------+
    // | print tpc data |
    // +----------------+
    Int_t pad_id = 0;
    TVector3 mom;
    reader.Restart();
    reader.SetEntry(event_id-1);
    while (reader.Next() ){
        if (event_id > 50) break;
        for(const auto& p_tpc : (*TPC)) {
            mom.SetXYZ( p_tpc.Px(), p_tpc.Py(), p_tpc.Pz() );
            pad_id = padHelper::findPadID(p_tpc.Vz(), p_tpc.Vx());
            if (0 <= pad_id && pad_id <= 5768) std::cout << event_id << "," << p_tpc.Vx() << "," << p_tpc.Vy() << "," << p_tpc.Vz() << "," << pad_id << std::endl;
        }
        event_id++;
    }

    // std::cout << "finish" << std::endl;
}

Int_t main(int argc, char** argv) { 
    TString path = argv[1];
    Int_t max_iter = 5;
    if (argc > 2) max_iter = atoi(argv[2]);
    analyze(path, max_iter);
    return 0;
}