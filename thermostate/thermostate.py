"""
Base ThermoState module
"""
from math import isclose
import sys

import CoolProp
from pint import UnitRegistry, DimensionalityError
from pint.unit import UnitsContainer, UnitDefinition
from pint.converters import ScaleConverter

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import numpy as np
try:  # pragma: no cover
    from IPython.core.ultratb import AutoFormattedTB
except ImportError:  # pragma: no cover
    AutoFormattedTB = None

units = UnitRegistry(autoconvert_offset_to_baseunit=True)
Q_ = units.Quantity
units.define(UnitDefinition('percent', 'pct', (), ScaleConverter(1.0/100.0)))

# Don't add the _render_traceback_ function to DimensionalityError if
# IPython isn't present. This function is only used by the IPython/ipykernel
# anyways, so it doesn't matter if it's missing if IPython isn't available.
if AutoFormattedTB is not None:  # pragma: no cover
    def render_traceback(self):
        """Render a minimized version of the DimensionalityError traceback

        The default Jupyter/IPython traceback includes a lot of
        context from within pint that actually raises the
        DimensionalityError. This context isn't really needed for
        this particular error, since the problem is almost certainly in
        the user code. This function removes the additional context.
        """
        a = AutoFormattedTB(mode='Context',
                            color_scheme='Neutral',
                            tb_offset=1)
        etype, evalue, tb = sys.exc_info()
        stb = a.structured_traceback(etype, evalue, tb, tb_offset=1)
        for i, line in enumerate(stb):
            if 'site-packages' in line:
                first_line = i
                break
        return stb[:first_line] + stb[-1:]

    DimensionalityError._render_traceback_ = render_traceback.__get__(DimensionalityError)

phase_map = {getattr(CoolProp, i): i.split('_')[1] for i in dir(CoolProp) if 'iphase' in i}


def munge_coolprop_input_prop(prop):
    prop = prop.replace('_INPUTS', '').replace('mass', '').replace('D', 'V')
    return prop.replace('Q', 'X').lower().replace('t', 'T')


def isclose_quant(a, b, *args, **kwargs):
    return isclose(a.magnitude, b.magnitude, *args, **kwargs)


class StateError(Exception):
    """Errors associated with setting the `State` object"""

    def _render_traceback_(self):  # pragma: no cover
        """Render a minimized version of the `StateError` traceback

        The default Jupyter/IPython traceback includes a lot of
        context from within `State` where the `StateError` is raised.
        This context isn't really needed, since the problem is almost certainly in
        the user code. This function removes the additional context.
        """
        if AutoFormattedTB is not None:
            a = AutoFormattedTB(mode='Context',
                                color_scheme='Neutral',
                                tb_offset=1)
            etype, evalue, tb = sys.exc_info()
            stb = a.structured_traceback(etype, evalue, tb, tb_offset=1)
            for i, line in enumerate(stb):
                if 'site-packages' in line:
                    first_line = i
                    break


class ProcessError(Exception):
    """Errors associated with the `Process` object"""

    def _render_traceback_(self):
        """Render a minimized version of the `ProcessError` traceback

        See `StateError` _render_traceback_ docstring for explanation.
        """
        if AutoFormattedTB is not None:
            a = AutoFormattedTB(mode='Context',
                                color_scheme='Neutral',
                                tb_offset=1)
            etype, evalue, tb = sys.exc_info()
            stb = a.structured_traceback(etype, evalue, tb, tb_offset=1)
            for i, line in enumerate(stb):
                if 'site-packages' in line:
                    first_line = i
                    break


class State(object):
    """Basic State manager for thermodyanmic states

    Parameters
    ----------
    substance : `str`
        One of the substances supported by CoolProp
    T : `pint.UnitRegistry.Quantity`
        Temperature
    p : `pint.UnitRegistry.Quantity`
        Pressure
    u : `pint.UnitRegistry.Quantity`
        Mass-specific internal energy
    s : `pint.UnitRegistry.Quantity`
        Mass-specific entropy
    v : `pint.UnitRegistry.Quantity`
        Mass-specific volume
    h : `pint.UnitRegistry.Quantity`
        Mass-specific enthalpy
    x : `pint.UnitRegistry.Quantity`
        Quality
    """
    _allowed_subs = ['AIR', 'AMMONIA', 'WATER', 'PROPANE', 'R134A', 'R22', 'ISOBUTANE',
                     'CARBONDIOXIDE', 'OXYGEN', 'NITROGEN']

    _all_pairs = [munge_coolprop_input_prop(k) for k in dir(CoolProp.constants)
                  if 'INPUTS' in k and 'molar' not in k]
    _all_pairs.extend([k[::-1] for k in _all_pairs])

    _unsupported_pairs = ['Tu', 'Th', 'us']
    _unsupported_pairs.extend([k[::-1] for k in _unsupported_pairs])

    # This weird lambda construct is necessary because _unsupported_pairs can't be accessed
    # inside the list comprehension because of the namespacing rules for class attributes.
    # Trying to set _allowed_pairs in the __init__ leads to infinite recursion because of
    # how we're messing with __setattr__.
    _allowed_pairs = (lambda x=_unsupported_pairs, y=_all_pairs: [p for p in y if p not in x])()

    _all_props = list('Tpvuhsx')

    _read_only_props = ['cp', 'cv', 'phase']

    _dimensions = {
        'T': UnitsContainer({'[temperature]': 1.0}),
        'p': UnitsContainer({'[mass]': 1.0, '[length]': -1.0, '[time]': -2.0}),
        'v': UnitsContainer({'[length]': 3.0, '[mass]': -1.0}),
        'u': UnitsContainer({'[length]': 2.0, '[time]': -2.0}),
        'h': UnitsContainer({'[length]': 2.0, '[time]': -2.0}),
        's': UnitsContainer({'[length]': 2.0, '[time]': -2.0, '[temperature]': -1.0}),
        'x': UnitsContainer({}),
    }

    _SI_units = {
        'T': 'kelvin',
        'p': 'pascal',
        'v': 'meter**3/kilogram',
        'u': 'joules/kilogram',
        'h': 'joules/kilogram',
        's': 'joules/(kilogram*kelvin)',
        'x': 'dimensionless',
        'cp': 'joules/(kilogram*kelvin)',
        'cv': 'joules/(kilogram*kelvin)',
        'phase': None,
    }

    def __setattr__(self, key, value):
        if key.startswith('_') or key == 'sub':
            object.__setattr__(self, key, value)
        elif key in self._allowed_pairs:
            self._check_dimensions(key, value)
            self._set_properties(key, value)
        elif key in self._unsupported_pairs:
            raise StateError("The pair of input properties entered ({}) isn't supported yet. "
                             "Sorry!".format(key))
        else:
            raise AttributeError('Unknown attribute {}'.format(key))

    def __getattr__(self, key):
        if key in self._all_props:
            return object.__getattribute__(self, '_' + key)
        elif key in self._all_pairs:
            val_0 = object.__getattribute__(self, '_' + key[0])
            val_1 = object.__getattribute__(self, '_' + key[1])
            return val_0, val_1
        elif key == 'phase':
            return object.__getattribute__(self, '_' + key)
        elif key in self._read_only_props:
            return object.__getattribute__(self, '_' + key)
        else:
            raise AttributeError("Unknown attribute {}".format(key))

    def __eq__(self, other):
        """Use any two independent and intensive properties to
        test for equality. Choose T and v because the EOS tends
        to be defined in terms of T and density.
        """
        if isclose_quant(other.T, self.T) and isclose_quant(other.v, self.v):
            return True

    def __le__(self, other):
        return NotImplemented

    def __lt__(self, other):
        return NotImplemented

    def __gt__(self, other):
        return NotImplemented

    def __ge__(self, other):
        return NotImplemented

    def __init__(self, substance, **kwargs):
        if substance.upper() in self._allowed_subs:
            self.sub = substance.upper()
        else:
            raise ValueError('{} is not an allowed substance. Choose one of {}.'.format(
                substance, self._allowed_subs,
            ))

        self._abstract_state = CoolProp.AbstractState("HEOS", self.sub)

        input_props = ''
        for arg in kwargs:
            if arg not in self._all_props:
                raise ValueError('The argument {} is not allowed.'.format(arg))
            else:
                input_props += arg

        if len(input_props) > 2 or len(input_props) == 1:
            raise ValueError('Incorrect number of properties specified. Must be 2 or 0.')

        if len(input_props) > 0 and input_props not in self._allowed_pairs:
            raise StateError("The pair of input properties entered ({}) isn't supported yet. "
                             "Sorry!".format(input_props))

        if len(input_props) > 0:
            setattr(self, input_props, (kwargs[input_props[0]], kwargs[input_props[1]]))

    def to_SI(self, prop, value):
        return value.to(self._SI_units[prop])

    def to_PropsSI(self, prop, value):
        return self.to_SI(prop, value).magnitude

    def _check_dimensions(self, properties, value):
        if value[0].dimensionality != self._dimensions[properties[0]]:
            raise StateError('The dimensions for {props[0]} must be {dim}'.format(
                props=properties,
                dim=self._dimensions[properties[0]]))
        elif value[1].dimensionality != self._dimensions[properties[1]]:
            raise StateError('The dimensions for {props[1]} must be {dim}'.format(
                props=properties,
                dim=self._dimensions[properties[1]]))

    def _set_properties(self, known_props, known_values):
        if len(known_props) != 2 or len(known_values) != 2 or len(known_props) != len(known_values):
            raise StateError('Only specify two properties to _set_properties')

        known_state = []

        for prop, val in zip(known_props, known_values):
            if prop == 'x':
                known_state.append(('Q', self.to_PropsSI(prop, val)))
            elif prop == 'v':
                known_state.append(('Dmass', 1.0/self.to_PropsSI(prop, val)))
            else:
                postfix = '' if prop in ['T', 'p'] else 'mass'
                known_state.append((prop.upper() + postfix, self.to_PropsSI(prop, val)))

        known_state.sort(key=lambda p: p[0])

        inputs = getattr(CoolProp, ''.join([p[0] for p in known_state]) + '_INPUTS')
        try:
            self._abstract_state.update(inputs, known_state[0][1], known_state[1][1])
        except ValueError as e:
            if 'Saturation pressure' in str(e):
                raise StateError('The given values for {} and {} are not '
                                 'independent.'.format(known_props[0], known_props[1]))
            else:
                raise

        for prop in self._all_props + self._read_only_props:
            if prop == 'v':
                value = Q_(1.0/self._abstract_state.keyed_output(CoolProp.iDmass),
                           self._SI_units[prop])
            elif prop == 'x':
                value = Q_(self._abstract_state.keyed_output(CoolProp.iQ), self._SI_units[prop])
                if value == -1.0:
                    value = None
            elif prop == 'phase':
                value = phase_map[self._abstract_state.keyed_output(CoolProp.iPhase)]
            else:
                postfix = '' if prop in ['T', 'p'] else 'mass'
                p = getattr(CoolProp, 'i' + prop.title() + postfix)
                value = Q_(self._abstract_state.keyed_output(p), self._SI_units[prop])

            setattr(self, '_' + prop, value)


class Process(object):
    """ This class allows for visualization of thermodynamic processes.
        Supports the following types of processes:
          - isobaric
          - isothermal
          - isentropic
          - isenthalpic

        Each process is defined by 3 aspects:
          - an initial state, a `State` object
          - a final state, a `State` object
          - a process type, a `str`

        Raises a `ProcessError` if the process_type cannot convert the
        initial state into the final state.
    """

    def __init__(self, state_1, state_2, process_type):
        self.state_1 = state_1
        self.state_2 = state_2
        self.sub = self.state_1.sub
        self.process_type = process_type

        self.validate()  # ensure process is possible

    def validate(self):
        """ Ensures the process type given can convert the initial state
            into the final state.
        """

        valid_processes = ["isobaric", "isothermal", "isentropic"]  # TODO add more process types

        process2property = {
            "isobaric": "pressure",
            "isothermal": "temperature",
            "isentropic": "specific entropy"
        }

        if self.process_type not in valid_processes:
            process_error_msg = ("\n The process type {} is not supported.\n"
                                 "Process must be one of: {}")

            raise ProcessError(process_error_msg.format(self.process_type, valid_processes))

        if ((self.process_type == "isobaric" and not(isclose_quant(self.state_1.p, self.state_2.p))) or
                (self.process_type == "isothermal" and not(isclose_quant(self.state_1.T, self.state_2.T))) or
                (self.process_type == "isentropic" and not(isclose_quant(self.state_1.s, self.state_2.s)))):

            property = process2property[self.process_type]

            if self.process_type == "isobaric":
                state_1_property_val = self.state_1.p
                state_2_property_val = self.state_2.p
            elif self.process_type == "isothermal":
                state_1_property_val = self.state_1.T
                state_2_property_val = self.state_2.T
            elif self.process_type == "isentropic":
                state_1_property_val = self.state_1.s
                state_2_property_val = self.state_2.s

            process_error_msg = ("\n{} process requires initial and final state to have the same {}\n"
                                 "State 1 {}: {}\n"
                                 "State 2 {}: {}")

            raise ProcessError(process_error_msg.format(self.process_type.capitalize(), property,
                                                        property, state_1_property_val,
                                                        property, state_2_property_val))

    def Ts_diagram(self, dome=True):
        """ Creates a MatplotLib graph of a temperature-specific entropy diagram.
            Temperature is on the vertical axis, specific entropy on the horizontal.
            If dome == True, then the vapor dome is also drawn on the diagram.

            Axes are scaled to one of the following:
                - +/- 10% of the max/min values of temperature/specific entropy
                - To show the top, bottom, left, and right sides of the vapor dome,
                  the process, and a 10% buffer on all sides depending on whether the
                  vapor dome or process is the topmost/rightmost/leftmost/bottommost
                  line on the graph.

            TODO: vapor dome support, state 1 and 2 point support
        """

        T_lo = min(self.state_1.T, self.state_2.T)
        T_hi = max(self.state_1.T, self.state_2.T)

        s_lo = min(self.state_1.s, self.state_2.s)
        s_hi = max(self.state_1.s, self.state_2.s)

        precision = 250  # how many data points from which a curve should be interpolated

        if self.process_type == "isobaric":
            specific_entropies = np.linspace(s_lo.magnitude, s_hi.magnitude, precision)
            pressure = self.state_1.p
            temperatures = []
            for specific_entropy in specific_entropies:
                state_i = State(self.sub, p=pressure, s=Q_(specific_entropy, self.state_1.s.units))
                T = state_i.T.magnitude
                temperatures.append(T)
        elif self.process_type == "isothermal":
            specific_entropies = np.linspace(s_lo.magnitude, s_hi.magnitude, precision)
            temperatures = [self.state_1.T.magnitude for i in range(precision)]
        elif self.process_type == "isentropic":
            specific_entropies = [self.state_1.s.magnitude for i in range(precision)]
            temperatures = np.linspace(T_lo.magnitude, T_hi.magnitude, precision)

        # set up plot
        fig, ax = plt.subplots()

        # plot vapor dome if applicable
        # done first so everything appears in front of vapor dome
        # TODO

        # plot process curve
        # done first so points appear on top of curve
        ax.plot(specific_entropies, temperatures, color="blue")

        # plot and label state points
        ax.plot(self.state_1.s.magnitude, self.state_1.T.magnitude, marker="o", color="red")
        ax.plot(self.state_2.s.magnitude, self.state_2.T.magnitude, marker="o", color="red")

        ax.annotate("1", (self.state_1.s.magnitude, self.state_1.T.magnitude))
        ax.annotate("2", (self.state_2.s.magnitude, self.state_2.T.magnitude))

        ax.set(xlabel='Specific Entropy ({})'.format(str(self.state_1.s.units)),
               ylabel='Temperature ({})'.format(str(self.state_1.T.units)),
               title='Temperature vs Specific Entropy ({} Process)'.format(self.process_type.capitalize()))
        ax.grid()

        plt.show()
