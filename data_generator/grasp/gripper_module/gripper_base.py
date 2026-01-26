import abc


class GripperBase(abc.ABC):
    """Base class for all grippers.
    Any gripper should subclass this class.
    You have to implement the following class method:
        - reset():
        - update_tcp_constraint():
        - set_tcp():
        - move_tcp_xyz():
        - detect_contact():
        - move():
        - read():
    """

    @abc.abstractmethod
    def reset(self):
        pass

    # @abc.abstractmethod
    # def update_tcp_constraint(self):
    #     pass

    @abc.abstractmethod
    def set_tcp(self):
        pass

    @abc.abstractmethod
    def move_tcp_xyz(self):
        pass

    @abc.abstractmethod
    def detect_contact(self):
        pass

    @abc.abstractmethod
    def move(self):
        pass

    @abc.abstractmethod
    def read(self):
        pass
