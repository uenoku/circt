
module FMA2(
    input  logic [10:0] a, b, c, d, e,
    output logic [30:0] y
);
    assign y = a * b + c * d * 2 + e + 1;
endmodule
