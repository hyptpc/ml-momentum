#include <iostream>
#include <sys/stat.h>
#include <fstream>
#include <vector>
#include <random>

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
#include "TLorentzVector.h"
#include "TRandom.h"

#include "include/padHelper.hh"

static const Double_t N_a = 6.02214076   * TMath::Power(10,  23);  // mol^-1
static const Double_t r_e = 2.8179403262 * TMath::Power(10, -13);  // cm
static const Double_t m_e = 0.5109989461;                        // MeV/c^2
static const Double_t m_p = 938.2720813;                         // MeV/c^2
static const Double_t e   = 1.60217662   * TMath::Power(10, -19);  // C
static const Double_t c   = 2.99792458   * TMath::Power(10,   8);  // m/s
static const Double_t rho = 0.0016755;                           // g/cm^3

Double_t bethe(TLorentzVector LV, Int_t Z, Int_t A, Int_t z, Double_t M = m_p){

    Double_t beta = LV.Beta();
    Double_t gamma = LV.Gamma();
    Double_t delta = 0;
    Double_t C = 0;
    Double_t I = (12*Z + 7) * TMath::Power(10, -6);
    if (13 <= Z) I = (9.76 + 58.8*TMath::Power(Z, -1.19) ) * Z * TMath::Power(10, -6);
    Double_t eta = beta*gamma;
    Double_t s = m_e / M;
    Double_t W_max = 2*m_e*eta*eta / ( 1 + 2*s*TMath::Sqrt( 1 + eta*eta ) + s*s );
    
    Double_t coefficient  = 2 * TMath::Pi() * N_a * r_e*r_e * m_e;
    Double_t correct_term = TMath::Log( 2 * m_e * gamma*gamma * beta*beta * W_max / (I*I) ) - 2*beta*beta - delta - 2*C/Z;

    Double_t dedx = coefficient * rho * Z/A * z*z / (beta*beta) * correct_term;

    return dedx;
}

void analyze(TString path, Int_t max_iter){

    // +-----------+
    // | load file |
    // +-----------+
    auto *f = new TFile(path.Data());

    TTreeReader reader("g4hyptpc", f);
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
    ofs << "evnum,x,y,z,pad_id,mom,px,py,pz,dE\n";

    // -- event selection and write --------------------------------
    Int_t pad_id = 0, counter = 0;
    Double_t dedx, dedx_bethe;
    TVector3 mom;
    TLorentzVector LV;
    reader.Restart();
    while (reader.Next() ){
        if (counter > max_iter) break;
        std::vector<Double_t> x;
        std::vector<Double_t> z;
        Double_t past_pad_id = -1;
        for(const auto& p_tpc : (*TPC)) {
            x.push_back(p_tpc.Vx());
            z.push_back(p_tpc.Vz());
            mom.SetXYZ( p_tpc.Px(), p_tpc.Py(), p_tpc.Pz() );
            LV.SetXYZM( p_tpc.Px(), p_tpc.Py(), p_tpc.Pz(), m_p );
            pad_id = padHelper::findPadID(p_tpc.Vz(), p_tpc.Vx());
            if (pad_id != past_pad_id) {
                dedx = bethe(LV, 18, 40, 1, m_p);
                // std::cout << counter << "," << p_tpc.Vx() << "," << p_tpc.Vy() << "," << p_tpc.Vz() << "," << pad_id << std::endl;
                // if (0 <= pad_id && pad_id <= 5768) ofs << counter << "," << p_tpc.Vx() << "," << p_tpc.Vy() << "," << p_tpc.Vz() << "," << pad_id << "," << mom.Mag() << "\n";
                if      (dedx > 0 &&    0 <= pad_id && pad_id <  1345) ofs << counter << "," << p_tpc.Vx() << "," << p_tpc.Vy() << "," << p_tpc.Vz() << "," << pad_id << "," << mom.Mag() << "," << p_tpc.Px() << "," << p_tpc.Py() << "," << p_tpc.Pz() << "," << dedx*0.9  << "\n";
                else if (dedx > 0 && 1345 <= pad_id && pad_id <= 5768) ofs << counter << "," << p_tpc.Vx() << "," << p_tpc.Vy() << "," << p_tpc.Vz() << "," << pad_id << "," << mom.Mag() << "," << p_tpc.Px() << "," << p_tpc.Py() << "," << p_tpc.Pz() << "," << dedx*1.25 << "\n";
                past_pad_id = pad_id;
            }
        }
        counter++;
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