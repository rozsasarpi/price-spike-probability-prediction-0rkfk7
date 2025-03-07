@echo off
REM Script for building Sphinx documentation on Windows

REM Configuration options for Sphinx
SET SPHINXOPTS=

REM Path to the sphinx-build executable
SET SPHINXBUILD=sphinx-build

REM Path to the sphinx-apidoc executable
SET SPHINXAPIDOC=sphinx-apidoc

REM Source directory containing documentation source files
SET SOURCEDIR=.

REM Directory where built documentation will be placed
SET BUILDDIR=_build

REM Directory where API documentation will be generated
SET APIDIR=source/api

REM Directory containing the Python package to document
SET PACKAGEDIR=..\src\backend

REM Display help information about available targets
:help
    echo.
    echo Please use ^(make ^<target^>) where ^<target^> is one of
    echo.
    echo   html       to make standalone HTML files
    echo   dirhtml    to make HTML files named index.html in each directory
    echo   singlehtml to make a single large HTML file
    echo   latex      to make LaTeX files, you can set SPHINXOPTS before
    echo            invoking make
    echo   latexpdf   to make LaTeX files and run them through pdflatex
    echo   text       to make text files
    echo   man        to make man pages
    echo   clean      to remove all generated files
    echo   linkcheck  to check all external links for integrity
    echo   doctest    to run all doctests
    echo   coverage   to check documentation coverage
    echo   apidoc     to generate API documentation
    echo.
    goto end

REM Build HTML documentation
:html
    echo Generating API documentation...
    call %SPHINXAPIDOC% -o %APIDIR% %PACKAGEDIR%
    if errorlevel 1 goto error

    echo Building HTML documentation...
    %SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %APIDIR%
    if errorlevel 1 goto error

    echo.
    echo HTML documentation successfully generated in %BUILDDIR%\html
    goto end

REM Build HTML documentation with directory structure
:dirhtml
    echo Generating API documentation...
    call %SPHINXAPIDOC% -o %APIDIR% %PACKAGEDIR%
    if errorlevel 1 goto error

    echo Building directory-based HTML documentation...
    %SPHINXBUILD% -M dirhtml %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %APIDIR%
    if errorlevel 1 goto error

    echo.
    echo Directory-based HTML documentation successfully generated in %BUILDDIR%\dirhtml
    goto end

REM Build single HTML file documentation
:singlehtml
    echo Generating API documentation...
    call %SPHINXAPIDOC% -o %APIDIR% %PACKAGEDIR%
    if errorlevel 1 goto error

    echo Building single HTML file documentation...
    %SPHINXBUILD% -M singlehtml %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %APIDIR%
    if errorlevel 1 goto error

    echo.
    echo Single HTML file documentation successfully generated in %BUILDDIR%\singlehtml
    goto end

REM Build LaTeX documentation for PDF generation
:latex
    echo Generating API documentation...
    call %SPHINXAPIDOC% -o %APIDIR% %PACKAGEDIR%
    if errorlevel 1 goto error

    echo Building LaTeX documentation...
    %SPHINXBUILD% -M latex %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %APIDIR%
    if errorlevel 1 goto error

    echo.
    echo LaTeX files successfully generated in %BUILDDIR%\latex
    goto end

REM Build PDF documentation via LaTeX
:latexpdf
    echo Generating LaTeX files...
    call make latex
    if errorlevel 1 goto error

    echo Building PDF documentation...
    cd %BUILDDIR%\latex
    make
    if errorlevel 1 goto error
    cd ..\..

    echo.
    echo PDF documentation successfully generated in %BUILDDIR%\latex
    goto end

REM Build plain text documentation
:text
    echo Generating API documentation...
    call %SPHINXAPIDOC% -o %APIDIR% %PACKAGEDIR%
    if errorlevel 1 goto error

    echo Building plain text documentation...
    %SPHINXBUILD% -M text %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %APIDIR%
    if errorlevel 1 goto error

    echo.
    echo Plain text documentation successfully generated in %BUILDDIR%\text
    goto end

REM Build manual pages
:man
    echo Generating API documentation...
    call %SPHINXAPIDOC% -o %APIDIR% %PACKAGEDIR%
    if errorlevel 1 goto error

    echo Building manual pages...
    %SPHINXBUILD% -M man %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %APIDIR%
    if errorlevel 1 goto error

    echo.
    echo Manual pages successfully generated in %BUILDDIR%\man
    goto end

REM Clean build directory by removing all generated files
:clean
    echo Cleaning build directory...
    if exist %BUILDDIR% (
        rmdir /s /q %BUILDDIR%
    )
    echo Build directory cleaned.
    goto end

REM Check all external links for integrity
:linkcheck
    echo Checking external links...
    %SPHINXBUILD% -b linkcheck %SOURCEDIR% %BUILDDIR%
    if errorlevel 1 goto error

    echo.
    echo External link check complete. See report in %BUILDDIR%\linkcheck
    goto end

REM Run doctests in the documentation
:doctest
    echo Running doctests...
    %SPHINXBUILD% -b doctest %SOURCEDIR% %BUILDDIR%
    if errorlevel 1 goto error

    echo.
    echo Doctests complete. See results in %BUILDDIR%\doctest
    goto end

REM Run coverage check on documentation
:coverage
    echo Running coverage check...
    %SPHINXBUILD% -b coverage %SOURCEDIR% %BUILDDIR%
    if errorlevel 1 goto error

    echo.
    echo Documentation coverage report generated in %BUILDDIR%\coverage
    goto end

REM Generate API documentation automatically from source code
:apidoc
    echo Generating API documentation...
    %SPHINXAPIDOC% -o %APIDIR% %PACKAGEDIR%
    if errorlevel 1 goto error

    echo.
    echo API documentation successfully generated in %APIDIR%
    goto end

:error
    echo An error occurred during the build process.
    exit /b 1

:end
    exit /b 0

REM Default target
if "%1"=="" goto help

REM Execute the specified target
if "%1"=="html" goto html
if "%1"=="dirhtml" goto dirhtml
if "%1"=="singlehtml" goto singlehtml
if "%1"=="latex" goto latex
if "%1"=="latexpdf" goto latexpdf
if "%1"=="text" goto text
if "%1"=="man" goto man
if "%1"=="clean" goto clean
if "%1"=="linkcheck" goto linkcheck
if "%1"=="doctest" goto doctest
if "%1"=="coverage" goto coverage
if "%1"=="apidoc" goto apidoc
goto help