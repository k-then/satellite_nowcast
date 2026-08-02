#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdint>
#include <memory>
#include "Vline_buffer_top.h"
#include "verilated.h"
#include "verilated_vcd_c.h"

constexpr int TILE_WIDTH  = 32;
constexpr int TILE_HEIGHT = 32;
constexpr int CHANNELS    = 2;
constexpr int DATA_WIDTH  = 16;

uint16_t float_to_fp16(float val) {
    return static_cast<uint16_t>(val * 65535.0f);
}

void tick(Vline_buffer_top* top, VerilatedVcdC* tfp, vluint64_t& main_time) {
    top->clk = 0;
    top->eval();
    if (tfp) tfp->dump(main_time);
    main_time += 5;

    top->clk = 1;
    top->eval();
    top->eval();  // extra eval to let combinational assigns fully settle
    if (tfp) tfp->dump(main_time);
    main_time += 5;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    auto top = std::make_unique<Vline_buffer_top>();

    Verilated::traceEverOn(true);
    auto tfp = std::make_unique<VerilatedVcdC>();
    top->trace(tfp.get(), 99);
    tfp->open("waveform.vcd");

    vluint64_t main_time = 0;

    // Reset Sequence
    top->rst = 1;
    top->s_axis_tvalid = 0;
    top->s_axis_tlast  = 0;
    top->m_axis_trdy   = 1; 
    top->s_axis_tdata  = 0;

    for (int i = 0; i < 5; ++i) {
        tick(top.get(), tfp.get(), main_time);
    }
    top->rst = 0;
    std::cout << "[TB] Reset released. Starting 2-Channel AXI-Stream Transfer...\n";

    std::vector<std::vector<uint16_t>> channel0_ir(TILE_HEIGHT, std::vector<uint16_t>(TILE_WIDTH));
    std::vector<std::vector<uint16_t>> channel1_radar(TILE_HEIGHT, std::vector<uint16_t>(TILE_WIDTH));

    for (int r = 0; r < TILE_HEIGHT; ++r) {
        for (int c = 0; c < TILE_WIDTH; ++c) {
            channel0_ir[r][c]    = static_cast<uint16_t>((r + 1) * 0x0100);
            channel1_radar[r][c] = static_cast<uint16_t>((c + 1) * 0x0001);
        }
    }

    int valid_window_count = 0;
    bool flushed_early = false;

    // DRIVE INPUT STREAM
    for (int r = 0; r < TILE_HEIGHT; ++r) {
        for (int c = 0; c < TILE_WIDTH; ++c) {
            
            uint32_t packed_tdata = (static_cast<uint32_t>(channel1_radar[r][c]) << 16) | 
                                    (static_cast<uint32_t>(channel0_ir[r][c]) & 0xFFFF);

            top->s_axis_tdata  = packed_tdata;
            top->s_axis_tvalid = 1;
            top->s_axis_tlast  = (r == TILE_HEIGHT - 1 && c == TILE_WIDTH - 1) ? 1 : 0;

            bool handshaked = false;

            while (!handshaked) {
                tick(top.get(), tfp.get(), main_time);

                // Sample valid output when downstream is ready
                if (top->m_axis_tvalid && top->m_axis_trdy) {
                    valid_window_count++;

                    // Catch tlast here too -- on the final pixel, the FSM can
                    // enter FLUSHING and run the entire flush sequence to
                    // completion *inside this same handshake loop*, before
                    // s_axis_trdy ever goes high again. Missing this check
                    // here means the flush pulse is silently skipped.
                    if (top->m_axis_tlast) {
                        flushed_early = true;
                    }

                    if (valid_window_count == 35) {
                        std::cout << "\n======================================================\n";
                        std::cout << " [TB] VALID WINDOW DETECTED (#" << valid_window_count << ")\n";
                        std::cout << "======================================================\n";
                        std::cout << " Channel 0 (IR) 3x3 Window:\n";
                        std::cout << "   [" << std::hex << std::setw(4) << top->ch0_win_00 << " " 
                                  << std::setw(4) << top->ch0_win_01 << " " 
                                  << std::setw(4) << top->ch0_win_02 << "]\n";
                        std::cout << "   [" << std::setw(4) << top->ch0_win_10 << " " 
                                  << std::setw(4) << top->ch0_win_11 << " " 
                                  << std::setw(4) << top->ch0_win_12 << "]\n";
                        std::cout << "   [" << std::setw(4) << top->ch0_win_20 << " " 
                                  << std::setw(4) << top->ch0_win_21 << " " 
                                  << std::setw(4) << top->ch0_win_22 << "]\n\n";

                        std::cout << " Channel 1 (Radar) 3x3 Window:\n";
                        std::cout << "   [" << std::setw(4) << top->ch1_win_00 << " " 
                                  << std::setw(4) << top->ch1_win_01 << " " 
                                  << std::setw(4) << top->ch1_win_02 << "]\n";
                        std::cout << "   [" << std::setw(4) << top->ch1_win_10 << " " 
                                  << std::setw(4) << top->ch1_win_11 << " " 
                                  << std::setw(4) << top->ch1_win_12 << "]\n";
                        std::cout << "   [" << std::setw(4) << top->ch1_win_20 << " " 
                                  << std::setw(4) << top->ch1_win_21 << " " 
                                  << std::setw(4) << top->ch1_win_22 << "]\n";
                        std::cout << "======================================================\n" << std::dec;
                    }
                }

                // Sample input ready AFTER clock edge
                if (top->s_axis_trdy) {
                    handshaked = true;
                }
            }
        }
    }

    // Clear input interface signals after active frame transfer completes
    top->s_axis_tvalid = 0;
    top->s_axis_tlast  = 0;
    top->s_axis_tdata  = 0;

    if (flushed_early) {
        std::cout << "\n[TB] tlast was already observed during the final pixel's "
                     "handshake -- flush completed inline, skipping dedicated flush loop.\n";
    } else {
        // Autonomous Flush Loop (only needed if tlast wasn't already caught above)
        bool flushed = false;
        int timeout = 0;

        while (!flushed) {
            tick(top.get(), tfp.get(), main_time);

            if (top->m_axis_tvalid && top->m_axis_trdy) {
                valid_window_count++;
            }

            if (top->m_axis_tlast && top->m_axis_tvalid && top->m_axis_trdy) {
                flushed = true;
            }

            // Watchdog to prevent terminal freeze
            if (++timeout > 5000) {
                std::cerr << "\n[TB ERROR] Flushing timed out! Check line_buffer_top FSM tlast logic.\n";
                break;
            }
        }
    }

    std::cout << "\n[TB] Simulation Complete!\n";
    std::cout << "[TB] Total Valid 3x3 Windows Produced: " << valid_window_count << "\n";

    if (tfp) tfp->close();
    top->final();
    return 0;
}