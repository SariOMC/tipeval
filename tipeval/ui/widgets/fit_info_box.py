import typing as T

from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QLabel, QWidget
from PyQt5 import uic

from tipeval.core.utils.data import get_resource_filename


if T.TYPE_CHECKING:
    from tipeval.core.tips import Tip, ThreeSidedPyramidTip


class PyramidTipInfoLayout(QWidget):

    def __init__(self, tip: 'ThreeSidedPyramidTip', *args, **kwargs):
        super().__init__(*args, **kwargs)

        import tipeval.ui.resources.ui_files

        with get_resource_filename(tipeval.ui.resources.ui_files, 'three_sided_pyramid_info_box.ui') as f:
            uic.loadUi(f, self)

        self._init_fields(tip)

    def _init_fields(self, tip: 'ThreeSidedPyramidTip') -> T.NoReturn:

        from tipeval.core.utils.nanoindentation import opening2cone_angle_three_sided_pyramid

        self.name_field.setText(tip.name)
        self.type_label.setText(tip.type)
        self.blunting_depth_label.setText(f'blunting depth ({tip.unit})')
        self.blunting_depth_field.setText(f'{tip.blunting_depth:.2f}')
        self.tilt_angle_field.setText(f'{tip.axis_inclination_angle:.2f}')
        self.equivalent_cone_angle_field.setText(f'{tip.equivalent_cone_angle:.2f} '
                                                 f'({opening2cone_angle_three_sided_pyramid(tip.ideal_angle):.2f})')

        for angle, field in zip(tip.angles_faces, [self.face_angle_field_1,
                                                   self.face_angle_field_2,
                                                   self.face_angle_field_3]):
            field.setText(f'{angle:.2f}')

        self.tip_angle_field.setText(f'{tip.tip_angle:.2f} ({tip.ideal_angle:.2f})')


TIP_INFO_LAYOUTS = {
    'Berkovich': PyramidTipInfoLayout,
    'Cube-corner': PyramidTipInfoLayout
}


class FitInfoBox(QGroupBox):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def set_tip(self, tip: 'Tip'):

        info_widget = TIP_INFO_LAYOUTS[tip.type](tip, self)
        self.layout().addWidget(info_widget)
