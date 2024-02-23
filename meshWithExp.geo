sigma = 0.05;
infx = 1.0;
infy = 0.5;
gamma = 1.0;
fend = sigma*Exp(-4./3.*Pi*gamma*infx^3);


Point(1) = {infx, fend, 0};
Point(2) = {infx, infy, 0};
Point(3) = {0,infy, 0};
Point(4) = {0, sigma, 0};

Point(5) = {sigma, fend, 0};

Line(11) = {1, 2};
Line(12) = {2, 3};
Line(13) = {3, 4};
BSpline(14) = {4, 5, 1};


Line Loop(15) = {11, 12, 13, 14};
Plane Surface(16) = {15};

Physical Surface(1) = {16};

//RIGHT
Physical Line(21) = {11};

//TOP
Physical Line(22) = {12};

//LEFT
Physical Line(23) = {13};

//CURVE
Physical Line(24) = {14};

Mesh.CharacteristicLengthMax=0.007;
Mesh.ScalingFactor = 1.0;
Mesh.Smoothing=10;
Mesh.Optimize = 10;

