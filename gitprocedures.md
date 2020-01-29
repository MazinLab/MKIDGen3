# Git Procedures
The UCSB Gen3 will use three different classes of git repositories: software repositories (e.g. for python
control code), Vivado HLS repositories, and Vivado project repositories.

## HLS Projects
    my-project-name/
      src/
        contains all source and testbench files
      data/ (optional)
        contains any data files
      my-project-name/  (in .gitignore)
        This is the HLS project that script.tcl will regenerate. Packaged IP will be placed in an archive inside in
        various solution directories. The copy of the exported IP will be seen by Vivado, though the IP Catalog will
        distinguish between versions if you properly set the version during export. This version should be deleted
        after extracting the exported zip file to the my-exported-ip-name/ directory in the repository top level. This 
        version MUST not be used in any Vivado projects that are committed to git.
      my-exported-ip-name/
        This directory contains a committed version of the packaged IP
      script.tcl
        This script, which may be built based on one from any existing project, will recreate the project and
        optionally simulate and package the ip: "vivado_hls -f script.tcl" 
      readme.md
        A readme file that describes the block
## Vivado Projects
    my-project-name/
      ip/
        some-ip-name/
          Each of these directories is either a git submodule (e.g. that of an HLS block) or of some other IP block
          (perhaps from a collaborator). If the IP block is to be used in more than one project it should be a
          submodule.
      bd/
        This is contains the top level block diagram exported with write_block_diagram (e.g. bd_2019.2.tcl). Note that
        projects can only be regenerated from the same version of Vivado that wrote the block diagram.
      top/
        This contains the vivado project and is excluded from git. The project should be configured to use 
        my-project-name/ip as its user IP repository.
      my-exported-ip-name/
        This optional directory should contain the packaged IP subsystem to enable the use of the project as IP in
        another vivado project. Vivado may want to export this to  my-project-name/ip/my-exported-ip-name which
        should also work. That export location may become the preferred location if this top-level location proves
        to me a manually executed hassle.
      proj.tcl
        This is a boilerplate file that is run to recreate the project. 
      readme.md
        A readme file that describes the block
## Software Projects
Procedures TBD, similar to MKIDPipeline/MKIDReadout