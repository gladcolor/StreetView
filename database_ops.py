from sqlalchemy import Column, String, Integer, Float,  create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.automap import automap_base
import sqlalchemy as db

from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    ForeignKey,
    DateTime,
    Sequence,
    Float,
    Table
)


engine = create_engine('sqlite:///databases//street_tree.db?check_same_thread=False', echo=True)
DB_Session = sessionmaker(bind=engine)
session = DB_Session()



Base = declarative_base()
# Base = automap_base()
# Base.prepare(engine, reflect=True)

# tables = Base.classes.keys()

# print("print(Base.classes.keys())：", Base.classes.keys())

# trees = Base.classes()
# keys =
#
# class Tree(Base):
#     __tablename__ = 'trees'
#
#     id             = Column(String(24), primary_key=True)
#     file_name      = Column(String(100))
#     longitude      = Column(Float)
#     latitude       = Column(Float)
#     elevation_wgs84_m = Column(Float)
#     distance       = Column(Float)
#
#     X              = Column(Float)
#     Y              = Column(Float)
#     H              = Column(Float)
#     GCS            = Column(String(60))
#
#     width_pixel    = Column(Float)
#     width_cm       = Column(Float)
#
#     width_mass_px  = Column(Float)
#     height_px      = Column(Float)
#
#     measure_row    = Column(Integer)
#     measure_col    = Column(Integer)
#
#     image_date     = Column(DateTime)
#     pano_id        = Column(String(30))
#     pano_yaw_deg   = Column(Float)
#     tilt_yaw_deg   = Column(Float)
#     tilt_pitch_deg = Column(Float)
#
#     theta          = Column(Float)
#     phi            = Column(Float)
#
#
#     GCS = Column(String(60))
#
#     def __repr__(self):
#         return "%s(%r)" % (self.__class__.__name__, self.username())



metadata = MetaData(engine)

tree_table = Table("tree_table", metadata, \
    Column('id', db.Integer, primary_key=True),
    Column('file_name', String(100)),
    Column('longitude', Float),
    Column("latitude", Float),
    Column("elevation_wgs84_m", Float),
    Column("distance", Float),

    Column("X", Float),
    Column("Y", Float),
    Column("H", Float),
    Column("GCS", String(60)),

    Column("width_pixel", Float),
    Column("width_cm", Float),

    Column("width_mass_px", Float),
    Column("height_px", Float),

    Column("measure_row", Integer),
    Column("measure_col", Integer),

    Column("image_date", DateTime),
    Column("pano_id", String(30)),
    Column("pano_yaw_deg", Float),
    Column("tilt_yaw_deg", Float),
    Column("tilt_pitch_deg", Float),

    Column("theta", Float),
    Column("phi", Float),

    Column("process_date", DateTime),

    __table_args__ = (db.UniqueConstraint('pano_id', 'phi', name="pano_phi"), \
                     db.Index("pano_id", 'phi', 'id', 'theta', 'file_name', 'longitude', 'latitude', 'distance', 'image_date')),    # combine index

    )

metadata.create_all()  # create table

# trees = Table('trees', metadata, autoload=True)

# print("trees" in metadata.tables)

class Tree(Base):
    __table__ = Table("tree_table", metadata, autoload=True)

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.pano_id())

    # def __repr__(self):
    #     return "%s(%r)" % (self.__class__.__name__, self.pano_id())
#
# print("type of trees:", type(Tree))
print([c.name for c in tree_table.columns])

# t1 = session.query(Tree).all()

# a_tree = Tree(pano_id="dfs", phi=3.343)
# session.add(a_tree)
# a_tree = Tree(pano_id="dfdsf", phi=3.343)
# session.add(a_tree)
# a_tree = Tree(pano_id="sdf", phi=3.343)
# session.add(a_tree)
# a_tree = Tree(pano_id="sdf", phi=3.343)
# session.add(a_tree)
a_tree = Tree(pano_id="test2", phi=3.1)
session.add(a_tree)

print(session.dirty)

session.commit()

# print(a_tree.pano_id)




# a_new_tree =




# sql = "show tables"
# session.execute(sql)