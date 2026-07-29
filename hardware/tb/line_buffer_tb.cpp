#include <iostream>
#include <memory>
#include <verilated.h>
#include <verilated_vcd_c.h>  // 1. Include VCD tracing header
#include "Vline_buffer_top.h"

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    
    // 2. Enable tracing in Verilator context
    Verilated::traceEverOn(true);

    auto top = std::make_unique<Vline_buffer_top>();
    auto tfp = std::make_unique<VerilatedVcdC>();

    // 3. Attach waveform tracer to top model
    top->trace(tfp.get(), 99); // Trace 99 levels of hierarchy (includes uut)
    tfp->open("waveform.vcd");

    vluint64_t sim_time = 0;

    // Initialize Signals
    top->clk = 0;
    top->rst = 1;
    top->s_axis_tvalid = 0;
    top->s_axis_tdata = 0;
    top->m_axis_trdy = 1;

    // Reset Sequence
    for (int i = 0; i < 10; i++) {
        top->clk = !top->clk;
        top->eval();
        tfp->dump(sim_time++);
    }
    top->rst = 0;

    std::cout << "--- Running Simulation & Generating waveform.vcd ---" << std::endl;

    int pixel_val = 1;
    for (int cycle = 0; cycle < 1500; cycle++) {
        // High Clock Edge
        top->clk = 1;
        top->s_axis_tvalid = 1;
        top->s_axis_tdata = pixel_val;
        top->eval();
        tfp->dump(sim_time++);

        if (top->s_axis_tvalid && top->s_axis_trdy) {
            pixel_val++;
        }

        // Low Clock Edge
        top->clk = 0;
        top->eval();
        tfp->dump(sim_time++);
    }

    top->final();
    tfp->close(); // 4. Flush and close waveform file

    std::cout << "Done! Waveform saved to waveform.vcd" << std::endl;
    return 0;
}