# CSCE 440: Final Project

## Introduction

This repository contains the code for our group's project, involving Lagrange interpolation of the Gaussian function within a library implementation of the Kuwahara filter.

## Structure

The client is a React application made up of one page. The source code can be found within the `client` directory.

The server is a Python Flask application with one API route. Additionally, our actual filter code can be found within `kuwahara.py`.

## How to Run

There are two parts to this codebase, as it is a web application: the client and the server.

### Client

- `cd` into `client/kuwahara-client`
- Run `npm i`
- Run `npm run start`

The client is now running. We can now start the server.

### Server

- `cd` into the `server/` directory.
- Run `python3 -m venv`. This creates a virtual environment where we can install the necessary packages!
- Depending on your operating system, do one of two things:
  - Windows: Run `venv/scripts/activate.bat` from `cmd` or `venv/scripts/activate.ps1` from `powershell`.
  - Linux/MacOS: Run `source venv/scripts/activate`.
  - For more information on `venv`, see [here](https://docs.python.org/3/library/venv.html)
- Run `pip install -r requirements.txt`
- Run `python3 ./server`

The server is now running as well. You should be able to interact with the web interface!
