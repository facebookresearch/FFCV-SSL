import numpy as np
import copy

from .utils import decode_null_terminated_string
from .types import (ALLOC_TABLE_TYPE, HeaderType, CURRENT_VERSION,
                    FieldDescType, get_handlers, get_metadata_type)

class Reader:
    def __init__(self, path, custom_handlers={}, custom_field_mapper=None):
        self._path = path
        self._custom_handlers = custom_handlers
        self._custom_field_mapper = custom_field_mapper
        self.read_header()
        self.read_field_descriptors()
        self.read_metadata()
        self.read_allocation_table()

    @property
    def file_name(self):
        return self._path

    def read_header(self):
        header = np.fromfile(self._path, dtype=HeaderType, count=1)[0]
        header.setflags(write=False)
        version = header["version"]

        if version != CURRENT_VERSION:
            msg = f"file format mismatch: code={CURRENT_VERSION},file={version}"
            raise AssertionError(msg)

        self.num_samples = header["num_samples"]
        self.page_size = header["page_size"]
        self.num_fields = header["num_fields"]
        self.header = header

    def read_field_descriptors(self):
        offset = HeaderType.itemsize
        field_descriptors = np.fromfile(
            self._path, dtype=FieldDescType, count=self.num_fields, offset=offset
        )
        field_descriptors.setflags(write=False)
        handlers = get_handlers(field_descriptors)

        self.field_descriptors = field_descriptors
        self.field_names = list(
            map(decode_null_terminated_string, self.field_descriptors["name"])
        )
        self.handlers = dict(zip(self.field_names, handlers))

        for field_name, field_desc in zip(self.field_names, self.field_descriptors):
            if field_name in self._custom_handlers:
                CustomHandler = self._custom_handlers[field_name]
                self.handlers[field_name] = CustomHandler.from_binary(
                    field_desc["arguments"]
                )
            elif self.handlers[field_name] is None:
                raise ValueError(
                    f"Must specify a custom_field entry "
                    f"for custom field {field_name}"
                )

        self.metadata_type = get_metadata_type(list(self.handlers.values()))

        if self._custom_field_mapper is not None:
            for target, source in self._custom_field_mapper.items():
                assert source in self.handlers
                self.handlers[target] = copy.copy(self.handlers[source])


    def read_metadata(self):
        offset = HeaderType.itemsize + self.field_descriptors.nbytes
        self.metadata = np.fromfile(
            self._path, dtype=self.metadata_type, count=self.num_samples, offset=offset
        )
        self.metadata.setflags(write=False)

    def read_allocation_table(self):
        offset = self.header["alloc_table_ptr"]
        alloc_table = np.fromfile(self._path, dtype=ALLOC_TABLE_TYPE, offset=offset)
        alloc_table.setflags(write=False)
        self.alloc_table = alloc_table