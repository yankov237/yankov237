from .core import ReferenceGenerator
from .core import PhysicalSystem
from .core import RewardFunction
from .core import ElectricMotorVisualization
from .core import ConstraintMonitor
from .constraints import Constraint, LimitConstraint
from .utils import make, register_superclass

register_superclass(RewardFunction)
register_superclass(ElectricMotorVisualization)
register_superclass(ReferenceGenerator)
register_superclass(PhysicalSystem)

import gym_electric_motor.reference_generators
import gym_electric_motor.reward_functions
import gym_electric_motor.visualization
import gym_electric_motor.physical_systems
import gym_electric_motor.envs
from gym.envs.registration import register

# Add all superclasses of the modules to the registry.


envs_path = 'gym_electric_motor.envs:'

# Version 1
register(id='Disc-SC-PermExDc-v0',
         entry_point=envs_path+'DiscSpeedControlDcPermanentlyExcitedMotorEnv')
register(id='Cont-SC-PermExDc-v0',
         entry_point=envs_path+'ContSpeedControlDcPermanentlyExcitedMotorEnv')
register(id='Disc-TC-PermExDc-v0',
         entry_point=envs_path+'DiscTorqueControlDcPermanentlyExcitedMotorEnv')
register(id='Cont-TC-PermExDc-v0',
         entry_point=envs_path+'ContTorqueControlDcPermanentlyExcitedMotorEnv')
register(id='Disc-CC-PermExDc-v0',
         entry_point=envs_path+'DiscCurrentControlDcPermanentlyExcitedMotorEnv')
register(id='Cont-CC-PermExDc-v0',
         entry_point=envs_path+'ContCurrentControlDcPermanentlyExcitedMotorEnv')

register(id='Disc-SC-ExtExDc-v0',
         entry_point=envs_path+'DiscSpeedControlDcExternallyExcitedMotorEnv')
register(id='Cont-SC-ExtExDc-v0',
         entry_point=envs_path+'ContSpeedControlDcExternallyExcitedMotorEnv')
register(id='Disc-TC-ExtExDc-v0',
         entry_point=envs_path+'DiscTorqueControlDcExternallyExcitedMotorEnv')
register(id='Cont-TC-ExtExDc-v0',
         entry_point=envs_path+'ContTorqueControlDcExternallyExcitedMotorEnv')
register(id='Disc-CC-ExtExDc-v0',
         entry_point=envs_path+'DiscCurrentControlDcExternallyExcitedMotorEnv')
register(id='Cont-CC-ExtExDc-v0',
         entry_point=envs_path+'ContCurrentControlDcExternallyExcitedMotorEnv')

register(id='DcSeriesCont-v1',
         entry_point=envs_path+'ContDcSeriesMotorEnvironment')
register(id='DcSeriesDisc-v1',
         entry_point=envs_path+'DiscDcSeriesMotorEnvironment')
register(id='DcShuntCont-v1',
         entry_point=envs_path+'ContDcShuntMotorEnvironment')
register(id='DcShuntDisc-v1',
         entry_point=envs_path+'DiscDcShuntMotorEnvironment')
register(id='PMSMCont-v1',
         entry_point=envs_path+'ContPermanentMagnetSynchronousMotorEnvironment')
register(id='PMSMDisc-v1',
         entry_point=envs_path+'DiscPermanentMagnetSynchronousMotorEnvironment')
register(id='SynRMCont-v1',
         entry_point=envs_path+'ContSynchronousReluctanceMotorEnvironment')
register(id='SynRMDisc-v1',
         entry_point=envs_path+'DiscSynchronousReluctanceMotorEnvironment')
register(id='SCIMCont-v1',
         entry_point=envs_path+'ContSquirrelCageInductionMotorEnvironment')
register(id='SCIMDisc-v1',
         entry_point=envs_path+'DiscSquirrelCageInductionMotorEnvironment')
register(id='DFIMCont-v1',
         entry_point=envs_path+'ContDoublyFedInductionMotorEnvironment')
register(id='DFIMDisc-v1',
         entry_point=envs_path+'DiscDoublyFedInductionMotorEnvironment')

# Old Ids

register(id='emotor-dc-extex-cont-v1',
         entry_point=envs_path+'ContDcExternallyExcitedMotorEnvironment')
register(id='emotor-dc-extex-disc-v1',
         entry_point=envs_path+'DiscDcExternallyExcitedMotorEnvironment')
register(id='emotor-dc-series-cont-v1',
         entry_point=envs_path+'ContDcSeriesMotorEnvironment')
register(id='emotor-dc-series-disc-v1',
         entry_point=envs_path+'DiscDcSeriesMotorEnvironment')
register(id='emotor-dc-shunt-cont-v1',
         entry_point=envs_path+'ContDcShuntMotorEnvironment')
register(id='emotor-dc-shunt-disc-v1',
         entry_point=envs_path+'DiscDcShuntMotorEnvironment')
register(id='emotor-pmsm-cont-v1',
         entry_point=envs_path+'ContPermanentMagnetSynchronousMotorEnvironment')
register(id='emotor-pmsm-disc-v1',
         entry_point=envs_path+'DiscPermanentMagnetSynchronousMotorEnvironment')
register(id='emotor-synrm-cont-v1',
         entry_point=envs_path+'ContSynchronousReluctanceMotorEnvironment')
register(id='emotor-synrm-disc-v1',
        entry_point=envs_path+'DiscSynchronousReluctanceMotorEnvironment')
register(id='emotor-scim-cont-v1',
         entry_point=envs_path+'ContSquirrelCageInductionMotorEnvironment')
register(id='emotor-scim-disc-v1',
         entry_point=envs_path+'DiscSquirrelCageInductionMotorEnvironment')
register(id='emotor-dfim-cont-v1',
         entry_point=envs_path+'ContDoublyFedInductionMotorEnvironment')
register(id='emotor-dfim-disc-v1',
         entry_point=envs_path+'DiscDoublyFedInductionMotorEnvironment')

