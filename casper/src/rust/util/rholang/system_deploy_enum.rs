// Enum wrapper for heterogeneous system deploys

use crate::rust::util::rholang::costacc::{close_block_deploy::CloseBlockDeploy, slash_deploy::SlashDeploy};

/// Enum to hold different types of system deploys in a homogeneous collection
#[derive(Clone)]
pub enum SystemDeployEnum {
    Slash(SlashDeploy),
    Close(CloseBlockDeploy),
}

impl SystemDeployEnum {
    pub fn as_slash(&self) -> Option<&SlashDeploy> {
        match self {
            SystemDeployEnum::Slash(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_close(&self) -> Option<&CloseBlockDeploy> {
        match self {
            SystemDeployEnum::Close(c) => Some(c),
            _ => None,
        }
    }

    pub fn as_slash_mut(&mut self) -> Option<&mut SlashDeploy> {
        match self {
            SystemDeployEnum::Slash(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_close_mut(&mut self) -> Option<&mut CloseBlockDeploy> {
        match self {
            SystemDeployEnum::Close(c) => Some(c),
            _ => None,
        }
    }
}

