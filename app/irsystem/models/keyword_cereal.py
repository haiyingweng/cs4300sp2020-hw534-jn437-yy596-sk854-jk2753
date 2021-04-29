from . import *


class KeywordCereal(db.Model):
    __tablename__ = "keyword_cereals"

    id = db.Column(db.Integer, primary_key=True)
    keyword = db.Column(db.String, nullable=False)
    cereal = db.Column(db.String, nullable=False)

    def __init__(self, **kwargs):
        self.keyword = kwargs.get("keyword", "")
        self.cereal = kwargs.get("cereal", "")

    def __repr__(self):
        return str(self.__dict__)


class KeywordCerealSchema(ModelSchema):
    class Meta:
        model = KeywordCereal
