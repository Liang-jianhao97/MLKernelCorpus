# MLKernelCorpus

This repo is an agentically built kernel corpus. 

## How to use

An end user can refresh the corpus using english. 

For example, to regenerate the corpus for MI450 kernels, the example usage is

1. Launch coding agent
2. Regenerate MI450 kernels following instructions: MI450/agent/generate_ir_corpus.md . Use docker container X

Where X is a docker container which has a compatible rocm/pytorch environment (e.g. one based on rocm/pytorch:latest).

## How it works 

The main sources of truth are

1. aiter
2. OAI manifest
3. Vanilla triton ML op support
4. Gluon ML op support

The user can either specify an agent to regenerate a manifest, or use the existing one.

Regenerate MI450 kernels following instructions: MI450/agent/generate_ir_corpus.md . Use docker container X

The user can either specify to regenerate manifests, or use the existing ones.

A manifest is a json file containing all relevant kernel locations + commands to run the kernel. 

It may be useful from time to time to regenerate the manifests, as this will take a snapshot of the current state of aiter + triton directories to attempt to find all relevant kernels and best configs.

On the other hand, using a stable manifest is more indicative of codegen drift caused by codegen changes, whereas regenerating may start tracking of different kernels.

To launch kernel with manifest regeneration, you must explicitly tell the agent to regenerate the manifests.

"Regenerate MI450 kernels following instructions: MI450/agent/generate_ir_corpus.md . Use docker container X. Before generating the kernels regenerate the manifests"

### aiter manifest generation

(aiter Repo + Fork specified in agents/forks.txt)

aiter specifies how the final AI models are constructed in terms of the ML ops. An agent will scrape the latest data from this specification to understand the most important kernels. 

Currently aiter uses multiple frameworks (FlyDSL, Triton, CK), but only triton kernel scraping is currently supported.

The agent will take the this understanding of what kernels are important and understand how to run them in triton (either via gluon if it is supported or vanilla triton if not). With this understanding, it will generate a manifest of kernel launches.

The subagent will not use the kernel implementations in the aiter branch itself. aiter is typically relatively far behind triton ToT, which means using a relatively old version of llvm. Looking this far back in time at llvm compilation is not helpful for driving current features.

### OAI manfiest

OAI manifest is manually defined by users. This is kernel location + command structure for launching kernels that OAI care about.

Currently, this only tracks decode + prefill FA.

### Vanilla triton & Gluon manifest generation

(Triton Repo + Fork specified in agents/forks.txt)

An agent will parse the triton source repo to undestand which ML ops are supported in vanilla triton and gluon. It will then generate a few commands for each kernel, specifying differnt kernel launch configs (e.g. different tile sizes). If available, it will always use num-warps=4 to specify 1 wave per eu. Once this investigation is done, it will generate a manifest opf kernel launches.

The agent will also try to look for hints for known best configs. Typically, we may find such information in tutorials/

## Generating the kernels

After launching the agent to regenerate the kernels, it will optionally generate the manifest. 

Then the agent will, in the container, 1. install triton and 2. take the manifests and use each to generate a cross compile command (with TRITON_OVERRIDE_ARCH).

Once done, it will repopulate the kernels in the kernels/ directory



