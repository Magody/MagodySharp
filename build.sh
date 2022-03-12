#!/bin/bash
framework="MagodySharp"
project_net_standard="MagodySharpNetStandard"
cp -r ./models/ dist/$project_net_standard
cd dist/$project_net_standard
dotnet build
mv ./bin/Debug/netstandard2.0/$project_net_standard.dll ../$framework
mv ./bin/Debug/netstandard2.0/*.dll ../$framework/dependencies
rm -rf ./models/ ./obj/ ./bin/
cd ../..